import unittest

from fastapi import HTTPException

from app.documents import RevisionConflict
from app.main import app
from app.routes.common import http_errors


class RouteTests(unittest.TestCase):
    def test_every_endpoint_is_registered_once(self) -> None:
        paths = app.openapi()["paths"]
        routes = {(method.upper(), path) for path, operations in paths.items() for method in operations}
        operation_ids = [operation["operationId"] for operations in paths.values() for operation in operations.values()]
        self.assertEqual(len(operation_ids), len(set(operation_ids)))
        expected = {
            ("POST", "/api/generate"), ("POST", "/api/edit"), ("POST", "/api/inpaint"), ("POST", "/api/outpaint"), ("POST", "/api/faces"),
            ("POST", "/api/select"), ("POST", "/api/select-text"), ("POST", "/api/cutout"),
            ("GET", "/api/jobs/{job_id}"), ("POST", "/api/jobs/{job_id}/cancel"),
            ("GET", "/api/history"), ("DELETE", "/api/history/{generation_id}"),
            ("GET", "/api/v2/documents"), ("PUT", "/api/v2/documents/{document_id}/layers"), ("POST", "/api/v2/documents/{document_id}/undo"),
            ("GET", "/api/v2/references"), ("POST", "/api/v2/references/{reference_id}/subject"),
            ("GET", "/api/engine"), ("POST", "/api/models/load"), ("POST", "/api/shutdown"),
        }
        self.assertLessEqual(expected, routes)

    def test_http_errors_map_the_most_specific_type_first(self) -> None:
        statuses = {LookupError: 404, RevisionConflict: 409, ValueError: 422}
        for error, status in ((KeyError("gone"), 404), (RevisionConflict("moved on"), 409), (ValueError("bad"), 422)):
            with self.subTest(error=error), self.assertRaises(HTTPException) as caught, http_errors(statuses):
                raise error
            self.assertEqual(caught.exception.status_code, status)
        with self.assertRaises(OSError), http_errors(statuses):
            raise OSError("not mapped")


if __name__ == "__main__":
    unittest.main()
