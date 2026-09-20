import os, sys
from fastapi.testclient import TestClient
BASE_DIR=os.path.normpath(os.path.join(os.path.dirname(__file__),".."));sys.path.insert(0,BASE_DIR)
from app.main import app, sanitize_filename
from app.config import settings
settings.ENVIRONMENT="test"
H={"Authorization":"Bearer test_token_hardening"}
def test_filename_sanitization():
    assert sanitize_filename("../../../etc/passwd")=="passwd"
    assert sanitize_filename("..\\..\\windows\\cmd.exe")=="cmd.exe"
def test_protected_document_upload_validates_type():
    with TestClient(app) as client:
        response=client.post("/api/knowledge/upload",headers=H,files={"file":("malicious.exe",b"x","application/octet-stream")})
        assert response.status_code==400
def test_invalid_token_is_rejected():
    with TestClient(app) as client:
        assert client.get("/api/machines",headers={"Authorization":"Bearer invalid"}).status_code==401
