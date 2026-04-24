import io
import os
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_factory.base import FileManager


class FakeResponse:
    def __init__(self, chunks=None, status_code=200, raise_error=None):
        self.chunks = chunks or []
        self.status_code = status_code
        self.raise_error = raise_error

    def raise_for_status(self):
        if self.raise_error is not None:
            raise self.raise_error

    def iter_content(self, chunk_size=1024):
        for chunk in self.chunks:
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestDataDownloadSafety(unittest.TestCase):
    def test_download_file_writes_atomically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "file.bin")
            with patch(
                "data_factory.base.requests.get",
                return_value=FakeResponse(chunks=[b"abc", b"def"]),
            ):
                FileManager.download_file("https://example.com/file.bin", save_path)

            with open(save_path, "rb") as handle:
                self.assertEqual(handle.read(), b"abcdef")

    def test_download_file_cleans_partial_temp_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "file.bin")
            error = RuntimeError("network down")
            with patch(
                "data_factory.base.requests.get",
                return_value=FakeResponse(raise_error=error),
            ):
                with self.assertRaises(RuntimeError):
                    FileManager.download_file("https://example.com/file.bin", save_path)

            self.assertFalse(os.path.exists(save_path))
            leftovers = [name for name in os.listdir(tmpdir) if name != "file.bin"]
            self.assertEqual(leftovers, [])

    def test_download_from_google_drive_checks_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "drive.bin")
            with patch("data_factory.base.gdown.download", return_value=None):
                with self.assertRaises(RuntimeError):
                    FileManager.download_from_google_drive("file-id", save_path)

            self.assertFalse(os.path.exists(save_path))

    def test_safe_extract_zip_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_bytes = io.BytesIO()
            with zipfile.ZipFile(zip_bytes, "w") as archive:
                archive.writestr("../escape.txt", "bad")

            with zipfile.ZipFile(io.BytesIO(zip_bytes.getvalue())) as archive:
                with self.assertRaises(ValueError):
                    FileManager._safe_extract_zip(archive, tmpdir)


if __name__ == "__main__":
    unittest.main()
