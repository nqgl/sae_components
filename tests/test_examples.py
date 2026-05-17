import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script_path", "expected_stdout"),
    [
        ("examples/define_architecture.py", "Architecture class: MyVanillaSAE"),
        ("examples/sweep_dsl.py", "1) Concrete config (no sweep)"),
    ],
)
def test_cpu_example_script_runs(script_path: str, expected_stdout: str) -> None:
    env = os.environ.copy()
    local_paths = [
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "sweepable" / "src"),
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        [*local_paths, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert expected_stdout in result.stdout
