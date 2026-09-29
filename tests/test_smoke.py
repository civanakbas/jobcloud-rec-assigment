from mlops_assignment.predict import load_bundle, predict_views
from mlops_assignment.train import train
from tests.support import sample_data


def test_train_and_predict(tmp_path):
    raw, companies = sample_data()
    postings_path = tmp_path / "postings.csv"
    companies_path = tmp_path / "company_industries.csv"
    raw.to_csv(postings_path, index=False)
    companies.to_csv(companies_path, index=False)

    out = tmp_path / "bundle.joblib"
    bundle = train(
        postings_path=str(postings_path),
        companies_path=str(companies_path),
        out_path=str(out),
    )
    assert out.exists()
    assert set(bundle["models"]) == {"views", "views_per_day"}

    loaded = load_bundle(str(out))
    sample = raw.head(5)
    preds_v = predict_views(sample, loaded, target="views")
    preds_r = predict_views(sample, loaded, target="views_per_day")
    assert len(preds_v) == len(sample) == len(preds_r)
    assert (preds_v >= 0).all() and (preds_r >= 0).all()
