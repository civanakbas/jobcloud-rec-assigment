import os
import subprocess
import sys


def test_installed_package_from_unrelated_directory(tmp_path):
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from importlib.metadata import version; "
            "from mlops_assignment.reports import evaluate; "
            "from mlops_assignment.ai_exercise.provider import ScriptedEnrichmentProvider; "
            "from mlops_assignment.enrichment.cli import run_curated; "
            "assert version('mlops-assignment') == '0.1.0'; "
            "assert callable(evaluate); "
            "assert len(run_curated()[0]) == 10",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
