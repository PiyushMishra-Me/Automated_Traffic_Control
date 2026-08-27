"""
__main__.py
Entrypoint for `python -m backend.decisionbackend` execution.
Runs the deterministic junction simulation and test validation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.decisionbackend.simulation import run_deterministic_simulation

if __name__ == "__main__":
    run_deterministic_simulation()
