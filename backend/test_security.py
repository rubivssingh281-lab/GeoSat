import io
from pathlib import Path
import pytest

from flask_backend import (
    app,
    get_data_dirs,
    resolve_safe_image_path,
    validate_image_magic_bytes,
    SlidingWindowRateLimiter,
    RATE_LIMITER,
)
from test_retrieval import safe_torch_load


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_path_traversal_protection():
    sar_dir, optical_dir = get_data_dirs()
    target_dir = optical_dir or sar_dir
    assert target_dir is not None

    # Test path traversal attempts
    assert resolve_safe_image_path(target_dir, "../../../etc/passwd") is None
    assert resolve_safe_image_path(target_dir, "..\\..\\windows\\win.ini") is None
    assert resolve_safe_image_path(target_dir, "test\x00.png") is None
    assert resolve_safe_image_path(target_dir, "subdir/image.png") is None


def test_path_traversal_api_rejection(client):
    res = client.get("/api/image/optical/..%2f..%2fsecret.txt")
    assert res.status_code == 404
    data = res.get_json()
    assert data["title"] == "Not Found"
    assert "error" in data


def test_magic_byte_validation():
    # Valid headers
    valid_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    valid_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
    valid_bmp = b"BM\x36\x00\x00\x00\x00\x00\x00\x00"

    assert validate_image_magic_bytes(valid_png) is True
    assert validate_image_magic_bytes(valid_jpeg) is True
    assert validate_image_magic_bytes(valid_bmp) is True

    # Invalid / spoofed header
    spoofed_text = b"<html><script>alert(1)</script></html>"
    assert validate_image_magic_bytes(spoofed_text) is False


def test_spoofed_image_upload_rejected_by_api(client):
    data = {
        "file": (io.BytesIO(b"fake text content not an image"), "payload.png"),
        "k": "6",
    }
    res = client.post("/api/retrieve", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    json_data = res.get_json()
    assert "magic byte signature" in json_data["error"]
    assert json_data["status"] == 400


def test_sliding_window_rate_limiter_unit():
    limiter = SlidingWindowRateLimiter(requests_per_minute=2, window_seconds=60.0)

    # 1st request allowed
    allowed, retry_after = limiter.is_allowed("192.168.1.100")
    assert allowed is True
    assert retry_after == 0

    # 2nd request allowed
    allowed, retry_after = limiter.is_allowed("192.168.1.100")
    assert allowed is True

    # 3rd request blocked
    allowed, retry_after = limiter.is_allowed("192.168.1.100")
    assert allowed is False
    assert retry_after >= 1

    # Different client IP is still allowed
    allowed_other, _ = limiter.is_allowed("192.168.1.101")
    assert allowed_other is True


def test_rate_limiter_triggers_429_on_api(client):
    old_limit = RATE_LIMITER.requests_per_minute
    RATE_LIMITER.requests_per_minute = 1
    RATE_LIMITER.reset()

    try:
        # First request (will trigger 400 due to missing query, but passes rate check)
        res1 = client.post("/api/retrieve", json={})
        assert res1.status_code == 400

        # Second rapid request exceeds rate limit of 1/minute
        res2 = client.post("/api/retrieve", json={})
        assert res2.status_code == 429
        assert "Retry-After" in res2.headers
        data = res2.get_json()
        assert data["status"] == 429
        assert "Too many requests" in data["error"]
        assert "retry_after" in data
    finally:
        RATE_LIMITER.requests_per_minute = old_limit
        RATE_LIMITER.reset()


def test_security_headers_present(client):
    res = client.get("/api/health")
    assert res.status_code == 200

    headers = res.headers
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "default-src 'self'" in headers.get("Content-Security-Policy", "")
    assert "camera=()" in headers.get("Permissions-Policy", "")
    assert "X-Request-ID" in headers
    assert len(headers["X-Request-ID"]) > 0


def test_rfc9457_error_details_format(client):
    res = client.get("/api/images/unsupported_kind")
    assert res.status_code == 400
    assert "application/problem+json" in res.headers.get("Content-Type", "")

    data = res.get_json()
    assert data["type"] == "about:blank"
    assert data["title"] == "Bad Request"
    assert data["status"] == 400
    assert "kind must be 'sar' or 'optical'" in data["detail"]
    assert data["instance"] == "/api/images/unsupported_kind"
    assert "request_id" in data
    assert "error" in data


def test_safe_torch_load_weights_only(monkeypatch):
    calls = []
    import torch

    orig_load = torch.load

    def mocked_load(*args, **kwargs):
        calls.append(kwargs)
        return orig_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", mocked_load)

    # Invoke safe_torch_load on an existing torch file
    weights_path = Path(__file__).resolve().parent / "saved_model" / "optical_embeddings.pt"
    if weights_path.exists():
        safe_torch_load(weights_path, map_location="cpu")
        assert len(calls) > 0
        assert calls[-1].get("weights_only") is True
