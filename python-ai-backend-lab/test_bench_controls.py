"""Prevent failed HTTP work and wrong aggregation from looking fast."""
import asyncio
import http.server
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/"bench"))
import bench_fanout as fan


class ErrorServer(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(503)
        self.end_headers()
        self.wfile.write(b"controlled failure")
    def log_message(self, *_):
        pass


class BenchmarkControls(unittest.TestCase):
    def test_failed_requests_cannot_count_as_successful_throughput(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1",0),ErrorServer)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        oldport = fan.PORT
        fan.PORT = server.server_port
        try:
            for method in (lambda: asyncio.run(fan.run_async(1,1)),
                           lambda: fan.run_threads(1,1), lambda: fan.run_procs(1,1)):
                with self.subTest(method=method), self.assertRaises(Exception):
                    method()
        finally:
            fan.PORT = oldport
            server.shutdown()
            server.server_close()

    def test_wrong_memory_aggregation_is_rejected(self):
        scratch = ROOT/"data/test-controls"
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            src = (ROOT/"bench/bench_mem.py").read_text()
            self.assertIn("total=sum(tot.values())",src)
            mutated = src.replace("total=sum(tot.values())", "total=sum(tot.values())+1")
            path = Path(tmp)/"broken_mem.py"
            path.write_text(mutated)
            env = dict(os.environ,PYTHONPATH=str(ROOT/"bench"),LAB_SCRATCH=tmp)
            result = subprocess.run([sys.executable,str(path),"--out",str(Path(tmp)/"out.json"),
                                     "--lines","1000","--repeats","1"],env=env,
                                    capture_output=True,text=True,timeout=20)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("AssertionError",result.stderr)


if __name__ == "__main__":
    unittest.main()
