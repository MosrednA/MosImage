from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class FileCacheTests(unittest.TestCase):
    def test_static_images_vary_by_origin_before_and_after_a_thumbnail_request(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", dir=settings.input_dir, delete=False) as temporary:
            temporary.write(b"test image content")
            path = Path(temporary.name)
        try:
            client = TestClient(app)
            url = "/files/images/inputs/" + path.name
            thumbnail = client.get(url)
            self.assertEqual(thumbnail.status_code, 200)
            self.assertIn("origin", thumbnail.headers["vary"].lower())
            opened = client.get(url, headers={"Origin": "http://127.0.0.1:5173"})
            self.assertEqual(opened.headers["access-control-allow-origin"], "http://127.0.0.1:5173")
            self.assertIn("origin", opened.headers["vary"].lower())
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
