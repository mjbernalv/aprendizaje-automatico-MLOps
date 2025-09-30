from mlops_example.data import load_boston_as_regression

def test_load_dataset():
    """
    Test loading the Boston Housing dataset.
    """
    X, y = load_boston_as_regression()
    assert len(X) == len(y)
    assert X.shape[0] > 100