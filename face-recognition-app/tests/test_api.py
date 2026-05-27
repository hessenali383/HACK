"""
Basic smoke tests for the FaceID Flask API.
These run without heavy ML deps (torch / facenet) so CI stays fast.
"""
import json
import os
import sys
import pytest

# Patch heavy imports before importing app
import unittest.mock as mock

# We mock torch and facenet so tests run without GPU deps
sys.modules['torch']                         = mock.MagicMock()
sys.modules['facenet_pytorch']               = mock.MagicMock()
sys.modules['sklearn']                       = mock.MagicMock()
sys.modules['sklearn.metrics']               = mock.MagicMock()
sys.modules['sklearn.metrics.pairwise']      = mock.MagicMock()
sys.modules['cv2']                           = mock.MagicMock()

# Set a temp DB path so tests don't touch production DB
os.environ['TESTING'] = '1'

from app import app, init_db, DB_PATH   # noqa: E402  (import after mocks)

# Use in-memory / temp DB for tests
import tempfile
_tmpdir = tempfile.mkdtemp()
import app as app_module
app_module.DB_PATH = os.path.join(_tmpdir, 'test_faces.db')


@pytest.fixture
def client():
    app.config['TESTING'] = True
    init_db()
    with app.test_client() as c:
        yield c


def test_index_returns_200(client):
    r = client.get('/')
    assert r.status_code == 200


def test_list_faces_empty(client):
    r = client.get('/api/faces')
    data = json.loads(r.data)
    assert 'faces' in data
    assert isinstance(data['faces'], list)


def test_delete_nonexistent_face(client):
    r = client.delete('/api/faces/99999')
    assert r.status_code == 200   # idempotent
    data = json.loads(r.data)
    assert data['success'] is True


def test_recognize_no_image(client):
    r = client.post('/api/recognize',
                    data=json.dumps({}),
                    content_type='application/json')
    # Should return 500 (missing key) — that's acceptable; no crash/500-loop
    assert r.status_code in (400, 500)


def test_save_face_no_image(client):
    r = client.post('/api/save-face',
                    data=json.dumps({'name': 'Test'}),
                    content_type='application/json')
    assert r.status_code in (400, 500)
