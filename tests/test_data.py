
from src.data import load_data


def test_dataset_is_not_empty():
    df = load_data()
    assert not df.empty


def test_target_is_binary():
    df = load_data()
    assert set(df["target"].unique()).issubset({0, 1})


def test_expected_columns_exist():
    df = load_data()
    assert "target" in df.columns
    assert len(df.columns) >= 14


def test_column_names_are_unique():
    df = load_data()
    assert df.columns.is_unique
