"""Phase 3 tests: the UI's ApiClient against the real FastAPI app (offline, via a sync ASGI transport)."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import cv2
import httpx
import numpy as np
import pytest

from api.app import create_app
from config import (
    AppConfig,
    AuthConfig,
    DatabaseConfig,
    DictionaryConfig,
    OCRConfig,
)
from db.repositories import DrugRepository, IngredientRepository, InteractionRepository
from ui.api_client import ApiClient, ApiError, ScanView


class SyncASGITransport(httpx.BaseTransport):
    """Runs an ASGI app synchronously on a fresh event loop per request.

    Newer httpx versions only provide an *async* ASGITransport, so the sync
    ApiClient used by Streamlit needs this bridge for offline testing.
    """

    def __init__(self, app) -> None:
        self.app = app

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": request.method,
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path,
            "query_string": request.url.query,
            "root_path": "",
            "headers": [(k.lower(), v) for k, v in request.headers.raw],  # ASGI requires lowercase names
            "client": ("testclient", 123),
            "server": (request.url.host or "testserver", request.url.port or 80),
        }
        request_body = request.read()
        status = {"code": 500}
        raw_headers: list[tuple[bytes, bytes]] = []
        chunks: list[bytes] = []

        async def receive() -> dict:
            return {"type": "http.request", "body": request_body, "more_body": False}

        async def send(message: dict) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                raw_headers.extend(message.get("headers", []))
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        asyncio.run(self.app(scope, receive, send))
        headers = [(k.decode("latin-1"), v.decode("latin-1")) for k, v in raw_headers]
        return httpx.Response(status["code"], headers=headers, content=b"".join(chunks))



def _make_app(tmp_path):
    """Same offline pattern as test_phase2_api: temp SQLite, OCR disabled, seeded drugs."""
    config = replace(
        AppConfig(),
        database=DatabaseConfig(url=f"sqlite:///{(tmp_path / 'ui.db').as_posix()}"),
        ocr=OCRConfig(enabled=False),
        auth=AuthConfig(secret_key="test-secret"),
        dictionary=DictionaryConfig(matching_dictionary_path=tmp_path / "match_dict.json"),
    )
    config.data.ensure_dirs()
    app = create_app(config)
    with app.state.database.session() as session:
        ingredients = IngredientRepository(session)
        interactions = InteractionRepository(session)
        drugs = DrugRepository(session)
        p = ingredients.add("paracetamol")
        i = ingredients.add("ibuprofen")
        interactions.add(p.id, i.id, "Major")
        drugs.add("paracetamol", ingredient_ids=[p.id])
        drugs.add("ibuprofen", ingredient_ids=[i.id])
        session.commit()
    return app


@pytest.fixture()
def api(tmp_path):
    transport = SyncASGITransport(app=_make_app(tmp_path))
    client = ApiClient("http://testserver", transport=transport)
    yield client
    client.close()


def _jpeg(path_tmp=None) -> bytes:
    image = np.full((480, 640, 3), 210, dtype=np.uint8)
    cv2.rectangle(image, (200, 150), (440, 330), (30, 30, 30), 10)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


class TestPatientFlow:
    def test_register_login_and_medication_roundtrip(self, api):
        token = api.register("Azhar", "azhar@example.com", "secret1")
        assert token
        api.set_token(token)
        assert api.list_medications() == []
        med = api.add_medication("paracetamol")
        assert med == {"drug_id": med["drug_id"], "drug_name": "paracetamol"}
        assert len(api.list_medications()) == 1
        api.remove_medication(med["drug_id"])
        assert api.list_medications() == []
        assert api.login("azhar@example.com", "secret1")

    def test_duplicate_register_and_unknown_drug_raise_api_error(self, api):
        api.register("Azhar", "azhar@example.com", "secret1")
        with pytest.raises(ApiError) as duplicate:
            api.register("Azhar", "azhar@example.com", "secret1")
        assert duplicate.value.status_code == 400
        api.set_token(api.login("azhar@example.com", "secret1"))
        with pytest.raises(ApiError) as unknown:
            api.add_medication("doesnotexist")
        assert unknown.value.status_code == 404

    def test_scan_and_history(self, api):
        api.set_token(api.register("Azhar", "azhar@example.com", "secret1"))
        view = api.scan_image(_jpeg(), "box.jpg")
        assert isinstance(view, ScanView)
        assert view.accepted and view.scan_id is not None
        history = api.scan_history()
        assert len(history) == 1
        assert history[0]["scan_id"] == view.scan_id
        assert history[0]["review_status"] == "auto"


class TestDoctorFlow:
    def test_drug_search_and_instant_check(self, api):
        names = [d["trade_name"] for d in api.search_drugs("PARA")]
        assert "paracetamol" in names
        report = api.doctor_check(["paracetamol", "ibuprofen"])
        assert report["status"] == "danger"
        assert report["findings"][0]["severity"] == "Major"

    def test_scan_returns_overlay_and_review_learns_dictionary(self, api, tmp_path):
        view = api.doctor_scan(_jpeg(), "box.jpg")
        assert view.accepted and view.scan_id is not None
        assert view.annotated_image_rgb is not None          # explainability overlay
        assert view.annotated_image_rgb.shape[2] == 3        # BGR→RGB decoded
        report = api.doctor_review(view.scan_id, "noisy ocr text", "paracetamol")
        assert report["status"] in {"safe", "caution", "danger"}
        dictionary = json.loads((tmp_path / "match_dict.json").read_text(encoding="utf-8"))
        assert "paracetamol" in dictionary.values()

    def test_review_unknown_drug_raises(self, api):
        view = api.doctor_scan(_jpeg(), "box.jpg")
        with pytest.raises(ApiError) as excinfo:
            api.doctor_review(view.scan_id, "noisy ocr text", "doesnotexist")
        assert excinfo.value.status_code == 404


class TestUnreachableServer:
    def test_network_error_maps_to_api_error(self):
        with ApiClient("http://127.0.0.1:59999", timeout=0.2) as client:
            with pytest.raises(ApiError) as excinfo:
                client.doctor_check(["paracetamol"])
        assert excinfo.value.status_code == 0
