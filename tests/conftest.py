import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import enhance_process as E  # noqa: E402,F401  (import-checks path works)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end tests that shell out to ffmpeg")
