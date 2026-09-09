"""HTTP client for the Streamlit UI — the single gateway to the FastAPI backend.

Uses httpx (the same stack as FastAPI's TestClient), so tests can drive the real
app through an ASGITransport without opening a socket.
"""
from __future__ import annotations

import base64

import cv2
import httpx
import numpy as np
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class ApiError(RuntimeError):
    """API or network failure with the HTTP status (0 = unreachable server)."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"[{status_code}] {detail}" if status_code else detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class ScanView:
    """UI-facing summary of a scan outcome (annotated image decoded on demand)."""

    accepted: bool = False
    rejection_reason: str = ""
    scan_id: int | None = None
    ocr_text: str = ""
    ocr_label: str = ""
    classifier_label: str = ""
    fused_label: str = ""
    fused_confidence: float = 0.0
    fusion_status: str = ""
    matched_drug_name: str = ""
    report: dict | None = None
    annotated_image_b64: str = ""

    @property
    def annotated_image_rgb(self) -> np.ndarray | None:
        if not self.annotated_image_b64:
            return None
        buffer = np.frombuffer(base64.b64decode(self.annotated_image_b64), dtype=np.uint8)
        image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # st.image expects RGB


class ApiClient:
    """Thin typed wrapper over the REST endpoints used by the two UI pages."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, token: str = "",
                 timeout: float = 180.0, transport: httpx.BaseTransport | None = None) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers,
                                    timeout=timeout, transport=transport)

    # -- lifecycle ------------------------------------------------------------
    def set_token(self, token: str) -> None:
        if token:
            self._client.headers["Authorization"] = f"Bearer {token}"
        else:
            self._client.headers.pop("Authorization", None)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ApiClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- plumbing -------------------------------------------------------------
    def _request(self, method: str, path: str, *, ok: tuple[int, ...] = (200, 201, 204),
                 **kwargs) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(0, f"Cannot reach the API server: {exc}") from exc
        if response.status_code not in ok:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:  # noqa: BLE001 — non-JSON error body
                detail = response.text
            raise ApiError(response.status_code, str(detail))
        return response

    # -- patient endpoints ----------------------------------------------------
    def register(self, name: str, email: str, password: str) -> str:
        response = self._request("POST", "/patient/register", json={
            "name": name, "email": email, "password": password,
        })
        return str(response.json()["access_token"])

    def login(self, email: str, password: str) -> str:
        response = self._request("POST", "/patient/login", json={
            "email": email, "password": password,
        })
        return str(response.json()["access_token"])

    def add_medication(self, drug_name: str) -> dict:
        return self._request("POST", "/patient/medications", json={"drug_name": drug_name}).json()

    def list_medications(self) -> list[dict]:
        return self._request("GET", "/patient/medications").json()

    def remove_medication(self, drug_id: int) -> None:
        self._request("DELETE", f"/patient/medications/{drug_id}")

    def scan_history(self) -> list[dict]:
        return self._request("GET", "/patient/scans").json()

    def scan_image(self, image_bytes: bytes, filename: str = "scan.jpg") -> ScanView:
        return self._scan("POST", "/scan/image", image_bytes, filename)

    # -- doctor/pharmacist endpoints (anonymous) ------------------------------
    def doctor_check(self, drug_names: list[str]) -> dict:
        return self._request("POST", "/doctor/check", json={"drug_names": drug_names}).json()

    def doctor_scan(self, image_bytes: bytes, filename: str = "scan.jpg") -> ScanView:
        return self._scan("POST", "/doctor/scan", image_bytes, filename)

    def doctor_review(self, scan_id: int, raw_ocr_text: str, corrected_drug_name: str,
                      corrected_by: str = "pharmacist") -> dict:
        return self._request("POST", "/doctor/review", json={
            "scan_id": scan_id,
            "raw_ocr_text": raw_ocr_text,
            "corrected_drug_name": corrected_drug_name,
            "corrected_by": corrected_by,
        }).json()

    def override_interaction(self, drug_a: str, drug_b: str, has_interaction: bool,
                             severity: str = "Moderate", description: str = "",
                             reviewed_by: str = "طبيب / صيدلاني") -> dict:
        return self._request("POST", "/doctor/interaction/override", json={
            "drug_a": drug_a,
            "drug_b": drug_b,
            "has_interaction": has_interaction,
            "severity": severity,
            "description": description,
            "reviewed_by": reviewed_by,
        }).json()

    def search_drugs(self, q: str = "", limit: int = 25) -> list[dict]:
        return self._request("GET", "/doctor/drugs",
                             params={"q": q, "limit": limit}).json()

    def recent_scans(self, limit: int = 30) -> list[dict]:
        return self._request("GET", "/doctor/recent-scans",
                             params={"limit": limit}).json()

    # -- shared ---------------------------------------------------------------
    def _scan(self, method: str, path: str, image_bytes: bytes, filename: str) -> ScanView:
        data = self._request(method, path,
                             files={"image": (filename, image_bytes, "image/jpeg")}).json()
        return ScanView(
            accepted=bool(data.get("accepted")),
            rejection_reason=data.get("rejection_reason", ""),
            scan_id=data.get("scan_id"),
            ocr_text=data.get("ocr_text", ""),
            ocr_label=data.get("ocr_label", ""),
            classifier_label=data.get("classifier_label", ""),
            fused_label=data.get("fused_label", ""),
            fused_confidence=float(data.get("fused_confidence", 0.0)),
            fusion_status=data.get("fusion_status", ""),
            matched_drug_name=data.get("matched_drug_name", ""),
            report=data.get("interaction_report"),
            annotated_image_b64=data.get("annotated_image_base64", ""),
        )
