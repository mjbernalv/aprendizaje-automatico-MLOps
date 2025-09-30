from src.mlops_example.modeling import build_model
from src.mlops_example.io_utils import save_model
import pandas as pd
import os

def test_dummy_predict(tmp_path):
    model = build_model("rf", n_estimators = 1)
    X = pd.DataFrame({"a": [0, 1, 2], "b": [0.1, 0.2, 0.3]})
    y = [0, 1, 2]
    model.fit(X, y)
    out = tmp_path / "m.joblib"
    save_model(model, str(out))
    assert os.path.exists(out)