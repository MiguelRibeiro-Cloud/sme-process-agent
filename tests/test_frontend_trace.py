import http.server
import shutil
import subprocess
import threading
import unittest
from functools import partial
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROME_PATHS = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


class FrontendTraceBrowserTests(unittest.TestCase):
    def test_demo_orientation_copy_is_present_without_policy_spoilers(self):
        html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="demo-guide"', html)
        self.assertEqual(html.count('class="prompt-suggestion"'), 4)
        self.assertIn("Northstar, a fictional industrial company", html)
        self.assertIn("documents contain synthetic data", html)
        self.assertIn("Interact with structured business systems and tools", html)
        self.assertIn("Search unstructured company knowledge semantically", html)
        self.assertIn("visitors cannot trigger paid evaluation runs", html)
        self.assertNotIn("Approval required above 10%", html)
        self.assertNotIn("Approval is required above 10%", html)

    def test_trace_persists_groups_toggles_and_resets_safely(self):
        chrome = next((path for path in CHROME_PATHS if path.exists()), None)
        if chrome is None:
            chrome_binary = shutil.which("google-chrome") or shutil.which("chromium")
            if chrome_binary:
                chrome = Path(chrome_binary)
        if chrome is None:
            self.skipTest("Chrome/Chromium is not installed")

        handler = partial(QuietHandler, directory=PROJECT_ROOT)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/tests/frontend_trace_harness.html"
            result = subprocess.run(
                [
                    str(chrome),
                    "--headless=new",
                    "--disable-gpu",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--virtual-time-budget=2000",
                    "--dump-dom",
                    url,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
        finally:
            server.shutdown()
            server.server_close()

        if result.returncode != 0 or "<html" not in result.stdout.lower():
            self.skipTest("Headless browser is unavailable in this execution environment")
        self.assertIn("<title>SMOKE:PASS</title>", result.stdout)


if __name__ == "__main__":
    unittest.main()
