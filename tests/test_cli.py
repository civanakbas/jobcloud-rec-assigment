"""Verify Fire preserves the documented executable interface."""

import subprocess
import sys

import pytest

from tests.support import sample_data


@pytest.mark.parametrize(
    "module",
    [
        "mlops_assignment.train",
        "mlops_assignment.validate",
        "mlops_assignment.enrichment.cli",
        "mlops_assignment.delivery.candidate",
        "mlops_assignment.delivery.release",
        "mlops_assignment.delivery.deploy",
    ],
)
def test_cli_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--", "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "SYNOPSIS" in result.stdout + result.stderr


def test_training_and_validation_flags(tmp_path):
    postings, companies = sample_data()
    postings_path = tmp_path / "postings.csv"
    companies_path = tmp_path / "companies.csv"
    artifact = tmp_path / "trained.joblib"
    postings.to_csv(postings_path, index=False)
    companies.to_csv(companies_path, index=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlops_assignment.train",
            "--postings",
            str(postings_path),
            "--companies",
            str(companies_path),
            "--out",
            str(artifact),
            "--test-size",
            "0.25",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert artifact.is_file()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlops_assignment.validate",
            "--candidate",
            str(artifact),
            "--incumbent",
            str(artifact),
            "--max-regression",
            "0.0",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert "candidate_mae" in result.stdout


def test_enrichment_requires_output_argument():
    result = subprocess.run(
        [sys.executable, "-m", "mlops_assignment.enrichment.cli"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "out" in result.stderr
