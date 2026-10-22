import subprocess
import sys
import time

COLD_START_LIMIT_SECONDS = 0.3
SERVE_IMPORTS = "import cachekeeper.adapters.inbound.cli, cachekeeper.adapters.inbound.http, uvicorn"


def test_cold_start_imports_stay_under_300ms() -> None:
    timings = []
    for _ in range(3):
        started = time.perf_counter()
        subprocess.run([sys.executable, "-c", SERVE_IMPORTS], check=True)  # noqa: S603 - fixed arguments
        timings.append(time.perf_counter() - started)
    assert min(timings) < COLD_START_LIMIT_SECONDS
