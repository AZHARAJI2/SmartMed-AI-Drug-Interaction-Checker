"""Phase 2 tests: auth security, AuthService, CorrectionLogger, and the API end-to-end."""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from auth.security import (
    HmacJwtCodec,
    Pbkdf2Hasher,
    build_password_hasher,
    build_token_codec,
)
from auth.service import AuthService
from config import (
    AppConfig,
    AuthConfig,
    DatabaseConfig,
    DictionaryConfig,
    OCRConfig,
)
from db.database import Database
from db.repositories import (
    DrugRepository,
    IngredientRepository,
    InteractionRepository,
    UserRepository,
)

# --------------------------------------------------------------------------- auth


class TestPasswordHashing:
    def test_roundtrip(self):
        hasher = Pbkdf2Hasher()
        hashed = hasher.hash("s3cret-password")
        assert hasher.verify("s3cret-password", hashed)
        assert not hasher.verify("wrong", hashed)

    def test_factory_hasher_verifies_own_hashes(self):
        hasher = build_password_hasher()
        hashed = hasher.hash("abc123")
        assert hasher.verify("abc123", hashed)

    def test_malformed_hash_is_false(self):
        assert not Pbkdf2Hasher().verify("x", "not-a-valid-hash")


class TestTokenCodec:
    def test_roundtrip(self):
        codec = HmacJwtCodec(AuthConfig())
        token = codec.encode({"sub": "1", "role": "patient", "exp": 9999999999})
        assert codec.decode(token)["sub"] == "1"

    def test_tampered_token_rejected(self):
        codec = HmacJwtCodec(AuthConfig())
        token = codec.encode({"sub": "1", "exp": 9999999999})
        with pytest.raises(ValueError):
            codec.decode(token + "x")

    def test_expired_token_rejected(self):
        codec = HmacJwtCodec(AuthConfig())
        token = codec.encode({"sub": "1", "exp": 1})
        with pytest.raises(ValueError):
            codec.decode(token)


class TestAuthService:
    def test_register_login_resolve(self, tmp_path):
        database = Database(DatabaseConfig(url=f"sqlite:///{(tmp_path / 'a.db').as_posix()}"))
        database.create_schema()
        config = AuthConfig()
        with database.session() as session:
            service = AuthService(
                UserRepository(session), build_password_hasher(), build_token_codec(config), config
            )
            service.register("Azhar", "Azhar@Example.com", "secret1")
            with pytest.raises(ValueError):
                service.register("Other", "azhar@example.com", "secret2")  # duplicate email
            user, token = service.authenticate("azhar@example.com", "secret1")
            assert user.name == "Azhar"
            resolved = service.resolve_token(token)
            assert resolved is not None and resolved.id == user.id
            assert service.resolve_token("garbage") is None
            with pytest.raises(ValueError):
                service.authenticate("azhar@example.com", "wrong")


# --------------------------------------------------------------------------- API helpers


def _make_app(tmp_path):
    config = replace(
        AppConfig(),
        database=DatabaseConfig(url=f"sqlite:///{(tmp_path / 'api.db').as_posix()}"),
        ocr=OCRConfig(enabled=False),  # offline-safe: fusion runs without OCR text
        auth=AuthConfig(secret_key="test-secret"),
        dictionary=DictionaryConfig(matching_dictionary_path=tmp_path / "match_dict.json"),
    )
    config.data.ensure_dirs()
    app = create_app(config)
    # seed the reference database: paracetamol + ibuprofen = Major interaction
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
def client(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as test_client:
        yield test_client


def _auth_header(client) -> dict:
    response = client.post("/patient/register", json={
        "name": "Azhar", "email": "azhar@example.com", "password": "secret1",
    })
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _jpeg(image: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", image)[1].tobytes()


class TestPatientEndpoints:
    def test_register_duplicate_rejected(self, client):
        _auth_header(client)
        response = client.post("/patient/register", json={
            "name": "Other", "email": "azhar@example.com", "password": "secret1",
        })
        assert response.status_code == 400

    def test_login_success_and_failure(self, client):
        _auth_header(client)
        ok = client.post("/patient/login", json={"email": "azhar@example.com", "password": "secret1"})
        assert ok.status_code == 200
        bad = client.post("/patient/login", json={"email": "azhar@example.com", "password": "nope"})
        assert bad.status_code == 401

    def test_medication_crud(self, client):
        headers = _auth_header(client)
        added = client.post("/patient/medications", json={"drug_name": "paracetamol"}, headers=headers)
        assert added.status_code == 201
        listed = client.get("/patient/medications", headers=headers)
        assert [m["drug_name"] for m in listed.json()] == ["paracetamol"]
        deleted = client.delete("/patient/medications/1", headers=headers)
        assert deleted.status_code == 204
        assert client.get("/patient/medications", headers=headers).json() == []

    def test_medication_requires_auth(self, client):
        assert client.get("/patient/medications").status_code == 401

    def test_unknown_drug_404(self, client):
        headers = _auth_header(client)
        response = client.post("/patient/medications", json={"drug_name": "zzz"}, headers=headers)
        assert response.status_code == 404

    def test_scan_history_empty(self, client):
        headers = _auth_header(client)
        assert client.get("/patient/scans", headers=headers).json() == []


class TestDoctorEndpoints:
    def test_text_check_danger(self, client):
        response = client.post("/doctor/check", json={"drug_names": ["paracetamol", "ibuprofen"]})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "danger"
        assert body["findings"][0]["severity"] == "Major"

    def test_text_check_single_safe(self, client):
        response = client.post("/doctor/check", json={"drug_names": ["paracetamol"]})
        assert response.json()["status"] == "safe"

    def test_compound_check(self, client):
        response = client.post("/doctor/check", json={"drug_names": ["paracetamol + ibuprofen"]})
        assert response.json()["status"] == "danger"


class TestScanEndpoints:
    def test_scan_image_flow(self, client):
        headers = _auth_header(client)
        image = np.full((480, 640, 3), 200, dtype=np.uint8)
        cv2.rectangle(image, (200, 150), (440, 330), (0, 0, 0), 8)
        response = client.post(
            "/scan/image",
            files={"image": ("pack.jpg", _jpeg(image), "image/jpeg")},
            headers=headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["accepted"] is True  # OCR disabled → no-signal fusion, but the pipeline runs
        history = client.get("/patient/scans", headers=headers).json()
        assert len(history) == 1

    def test_scan_rejects_poor_image(self, client):
        headers = _auth_header(client)
        response = client.post(
            "/scan/image",
            files={"image": ("dark.jpg", _jpeg(np.zeros((480, 640, 3), dtype=np.uint8)), "image/jpeg")},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json()["accepted"] is False

    def test_doctor_scan_anonymous(self, client):
        image = np.full((480, 640, 3), 200, dtype=np.uint8)
        cv2.rectangle(image, (200, 150), (440, 330), (0, 0, 0), 8)  # edges → passes blur gate
        response = client.post(
            "/doctor/scan",
            files={"image": ("p.jpg", _jpeg(image), "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["accepted"] is True

    def test_doctor_review_updates_dictionary_and_log(self, client, tmp_path):
        headers = _auth_header(client)
        image = np.full((480, 640, 3), 200, dtype=np.uint8)
        cv2.rectangle(image, (200, 150), (440, 330), (0, 0, 0), 8)  # edges → passes blur gate
        client.post("/scan/image", files={"image": ("p.jpg", _jpeg(image), "image/jpeg")}, headers=headers)
        scan_id = client.get("/patient/scans", headers=headers).json()[0]["scan_id"]

        response = client.post("/doctor/review", json={
            "scan_id": scan_id,
            "raw_ocr_text": "practmol 500",
            "corrected_drug_name": "paracetamol",
            "corrected_by": "pharmacist1",
        })
        assert response.status_code == 200

        # the review is recorded in the DB and the dictionary file learned the alias
        from correction_learning.logger import CorrectionLogger
        from db.repositories import CorrectionLogRepository, DrugRepository

        app = client.app
        with app.state.database.session() as session:
            assert len(CorrectionLogRepository(session).all()) == 1
            logger = CorrectionLogger(app.state.config.dictionary,
                                      CorrectionLogRepository(session), DrugRepository(session))
            dictionary = logger.load_dictionary()
        assert dictionary.get("practmol 500") == "paracetamol"

        # scan history now shows the corrected status
        history = client.get("/patient/scans", headers=headers).json()
        assert history[0]["review_status"] == "corrected"

    def test_review_unknown_drug_404(self, client):
        headers = _auth_header(client)
        response = client.post("/doctor/review", json={
            "scan_id": 1,
            "raw_ocr_text": "whatever",
            "corrected_drug_name": "doesnotexist",
        })
        assert response.status_code == 404

    def test_doctor_interaction_override(self, client):
        # 1. Override: set interaction between Paracetamol and Warfarin to Major
        res1 = client.post("/doctor/interaction/override", json={
            "drug_a": "paracetamol",
            "drug_b": "warfarin",
            "has_interaction": True,
            "severity": "Major",
            "description": "Risk of bleeding on prolonged use",
            "reviewed_by": "Dr. Azhar",
        })
        assert res1.status_code == 200
        data1 = res1.json()
        assert len(data1["findings"]) >= 1
        assert any(f["severity"] == "Major" for f in data1["findings"])

        # 2. Override: remove interaction (set has_interaction=False)
        res2 = client.post("/doctor/interaction/override", json={
            "drug_a": "paracetamol",
            "drug_b": "warfarin",
            "has_interaction": False,
            "reviewed_by": "Dr. Azhar",
        })
        assert res2.status_code == 200
        data2 = res2.json()
        assert len(data2["findings"]) == 0


