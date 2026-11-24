import io
import sys
import unittest
from pathlib import Path
from PIL import Image
import torch

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import flask_backend

class TestGeoSatIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = flask_backend.app.test_client()

    def test_01_health_check(self):
        resp = self.client.get('/api/health')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "ok")
        print(f"[TEST] Health: {data}")

    def test_02_status_check(self):
        resp = self.client.get('/api/status')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("dataset_found"))
        self.assertTrue(data.get("database_connected"))
        self.assertGreaterEqual(data.get("optical_images"), 30)
        print(f"[TEST] Status: Connected={data.get('database_connected')}, Images={data.get('optical_images')}, Device={data.get('device')}")

    def test_03_metrics_check(self):
        resp = self.client.get('/api/metrics')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("total_queries", data)
        print(f"[TEST] Metrics: {data}")

    def test_04_sar_to_optical_retrieval(self):
        # Create a synthetic 224x224 SAR query image
        img = Image.new('L', (224, 224), color=120)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        data = {
            'file': (buf, 'synthetic_sar_query.png'),
            'k': 5,
            'limit': 30,
            'refresh_cache': False
        }
        resp = self.client.post('/api/retrieve', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body.get("count"), 5)
        self.assertEqual(len(body.get("results")), 5)
        print(f"[TEST] Retrieval Succeeded! Latency: {body.get('latency_ms')}ms")
        for res in body.get("results"):
            print(f"       Rank {res['rank']}: {res['filename']} -> Score: {res['score']:.4f} ({res['match_percent']}%)")

    def test_05_serve_optical_image(self):
        # Fetch status to get first optical image
        resp = self.client.get('/api/images/optical')
        self.assertEqual(resp.status_code, 200)
        files = resp.get_json().get("files", [])
        self.assertGreater(len(files), 0)

        # Test image serving endpoint
        img_resp = self.client.get(f"/api/image/optical/{files[0]}")
        self.assertEqual(img_resp.status_code, 200)
        self.assertIn("image", img_resp.headers.get("Content-Type", ""))
        print(f"[TEST] Serve Image: {files[0]} returned {img_resp.status_code} ({len(img_resp.data)} bytes)")

if __name__ == "__main__":
    unittest.main()
