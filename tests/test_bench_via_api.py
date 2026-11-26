import io
import json
import os
import runpy
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import requests


CLIENT = Path(__file__).resolve().parents[1] / "clients" / "bench_via_api.py"
API = "https://parser.example.invalid/v3/parse"
RESULT_URL = "https://files.example.invalid/benchmark.json?signature=example"
DONE = {
    "job_id": "job-1",
    "status": "done",
    "pages": [
        {"page": 1, "markdown": "First page"},
        {"page": 2, "markdown": "Second page"},
    ],
}


# --- Offline CLI test helpers ---
def response(payload, status=200):
    result = requests.Response()
    result.status_code = status
    result.url = RESULT_URL
    result._content = json.dumps(payload).encode()
    return result


class BenchmarkClientTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        input_file = self.root / "pdfs" / "tables" / "fixture.pdf"
        input_file.parent.mkdir(parents=True)
        input_file.write_bytes(b"%PDF-1.4\n")
        self.output = self.root / "candidate" / "tables"

    def run_client(self, terminal, download=None):
        """Run the real CLI with a one-file dataset and offline transport."""
        submitted = response({"job_id": "job-1", "status": "queued"}, 201)
        polling = [submitted, response({"status": "running"}), response(terminal)]
        stdout = io.StringIO()
        with (
            patch.dict(os.environ, {"API": API, "KEY": "test-api-key"}),
            patch("sys.argv", [str(CLIENT), str(self.root), "candidate", "--parallel", "1"]),
            patch("time.sleep"),
            patch("requests.request", side_effect=polling) as jobs,
            patch("requests.get", return_value=download if not isinstance(download, Exception) else None,
                  side_effect=download if isinstance(download, Exception) else None) as fetch,
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as exited,
        ):
            runpy.run_path(str(CLIENT), run_name="__main__")
        self.assertEqual([call.args for call in jobs.call_args_list],
                         [("POST", API), ("GET", f"{API}/job-1"), ("GET", f"{API}/job-1")])
        self.assertTrue(all(call.kwargs["headers"] == {"X-API-Key": "test-api-key"} for call in jobs.call_args_list))
        return exited.exception.code, fetch, stdout.getvalue()

    def assert_pages_written(self):
        self.assertEqual(
            {path.name: path.read_text() for path in self.output.glob("*.md")},
            {"fixture_pg1_repeat1.md": "First page", "fixture_pg2_repeat1.md": "Second page"},
        )

    # --- End offline CLI test helpers ---

    def test_inline_result_writes_pages(self):
        exit_code, fetch, _ = self.run_client(DONE)
        self.assertEqual(exit_code, 0)
        self.assert_pages_written()
        fetch.assert_not_called()

    def test_result_url_writes_all_pages_without_submitting_again(self):
        terminal = {"job_id": "job-1", "status": "done", "result_url": RESULT_URL}
        exit_code, fetch, _ = self.run_client(terminal, response(DONE))
        self.assertEqual(exit_code, 0)
        self.assert_pages_written()
        fetch.assert_called_once_with(RESULT_URL, timeout=60)

    def test_result_download_http_failure_fails_cli(self):
        terminal = {"job_id": "job-1", "status": "done", "result_url": RESULT_URL}
        exit_code, _, output = self.run_client(terminal, response({"error": "expired"}, 403))
        self.assertEqual(exit_code, 1)
        self.assertIn("http-error:403", output)
        self.assertFalse(self.output.exists())

    def test_result_download_error_excludes_signed_url(self):
        terminal = {"job_id": "job-1", "status": "done", "result_url": RESULT_URL}
        exit_code, _, output = self.run_client(terminal, requests.ConnectionError(RESULT_URL))
        self.assertEqual(exit_code, 1)
        self.assertIn("ConnectionError", output)
        self.assertNotIn(RESULT_URL, output)

    def test_incomplete_downloaded_result_fails_cli(self):
        terminal = {"job_id": "job-1", "status": "done", "result_url": RESULT_URL}
        exit_code, _, _ = self.run_client(terminal, response({"status": "done"}))
        self.assertEqual(exit_code, 1)
        self.assertFalse(self.output.exists())

    def test_failed_parser_job_preserves_blank_stub(self):
        exit_code, fetch, _ = self.run_client({"status": "failed", "error": "test parser error"})
        self.assertEqual(exit_code, 1)
        self.assertEqual((self.output / "fixture_pg1_repeat1.md").read_text(), "")
        fetch.assert_not_called()
