"""Generate a trusted model from deterministic synthetic data for CI smoke checks."""

from pathlib import Path
from tempfile import TemporaryDirectory

import fire
import numpy as np
import pandas as pd

from mlops_assignment.train import train


def sample_data(row_count: int = 1200) -> tuple[pd.DataFrame, pd.DataFrame]:
    row = np.arange(row_count)
    listed_time = 1_713_000_000_000 + row * 60_000
    postings = pd.DataFrame(
        {
            "job_id": row,
            "company_id": row % 20,
            "title": [f"engineer skill{i % 30} sector{(i // 30) % 30}" for i in row],
            "description": [f"Build reliable system {i}" for i in row],
            "skills_desc": ["python" if i % 3 else None for i in row],
            "normalized_salary": 70_000 + (row % 80) * 1_000,
            "remote_allowed": row % 2,
            "sponsored": row % 5 == 0,
            "formatted_work_type": np.where(row % 4, "Full-time", "Contract"),
            "formatted_experience_level": np.where(row % 3, "Mid-Senior level", "Entry level"),
            "pay_period": "YEARLY",
            "application_type": np.where(row % 2, "OffsiteApply", "SimpleOnsiteApply"),
            "original_listed_time": listed_time,
            "listed_time": listed_time,
            "expiry": listed_time + 30 * 86_400_000,
            "views": 20 + (row % 100),
        }
    )
    companies = pd.DataFrame(
        {
            "company_id": np.arange(20),
            "industry": [f"industry-{i % 4}" for i in range(20)],
        }
    )
    return postings, companies


def generate_model(out: str = "artifacts/synthetic.joblib") -> None:
    """Train a smoke-test artifact without the full dataset or supplied model."""
    destination = Path(out).resolve()
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        postings, companies = sample_data()
        postings.to_csv(directory / "postings.csv", index=False)
        companies.to_csv(directory / "companies.csv", index=False)
        train(str(directory / "postings.csv"), str(directory / "companies.csv"), str(destination))
    # Docker's non-root user needs read access to this disposable test artifact.
    destination.chmod(0o644)
    print(destination)


if __name__ == "__main__":
    fire.Fire(generate_model)
