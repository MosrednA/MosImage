import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app, read_uploaded_image


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

    def test_database_and_non_image_files_are_not_served(self) -> None:
        client = TestClient(app)
        response = client.get("/files/app.db")
        self.assertEqual(response.status_code, 404)

    def test_upload_exceeding_size_limit_is_rejected(self) -> None:
        client = TestClient(app)
        oversized = b"x" * 1024
        with patch.object(settings, "max_upload_bytes", 500):
            response = client.post(
                "/api/v2/documents",
                data={"name": "Big"},
                files={"image": ("big.png", oversized, "image/png")},
            )
            self.assertEqual(response.status_code, 413)
            self.assertIn("exceeds maximum size", response.json()["detail"])


class UploadReadLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_upload_is_rejected_without_reading_the_entire_file(self) -> None:
        with io.BytesIO(b"x" * 10_000) as stream, patch.object(settings, "max_upload_bytes", 500):
            upload = UploadFile(file=stream, filename="large.png")
            with self.assertRaises(HTTPException) as raised:
                await read_uploaded_image(upload)
            self.assertEqual(raised.exception.status_code, 413)
            self.assertLessEqual(stream.tell(), 501)

    async def test_upload_at_the_limit_is_preserved(self) -> None:
        content = b"x" * 500
        with io.BytesIO(content) as stream, patch.object(settings, "max_upload_bytes", 500):
            self.assertEqual(await read_uploaded_image(UploadFile(file=stream, filename="image.png")), content)


if __name__ == "__main__":
    unittest.main()
