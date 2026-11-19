import pandas as pd
import pytest

from agent import analytics as A


@pytest.fixture(scope="module")
def df():
    return A.load_data()


def test_overview(df):
    o = A.dataset_overview(df)
    assert o["records"] == 10_000 and o["failures"] == 339
    assert o["failure_rate_percent"] == 3.39
    assert o["failure_mode_counts"] == {"TWF": 46, "HDF": 115, "PWF": 95, "OSF": 98, "RNF": 19}
    assert sum(o["product_types"].values()) == 10_000


def test_count_with_filters(df):
    assert A.count_machines(df, failed=True)["count"] == 339
    assert A.count_machines(df, failed=True, product_type="M")["count"] == \
        int(((df["Machine failure"] == 1) & (df["Type"] == "M")).sum())
    assert A.count_machines(df, failure_mode="HDF")["count"] == 115


def test_failure_rate_by_type(df):
    r = A.failure_rate(df, product_type="L")
    expected = round(100 * df[df["Type"] == "L"]["Machine failure"].mean(), 2)
    assert r["failure_rate_percent"] == expected
    assert set(r["failure_rate_percent_by_type"]) == {"L", "M", "H"}


def test_sensor_statistic_matches_pandas(df):
    r = A.sensor_statistic(df, "torque", "mean", failed=True)
    assert r["value"] == round(df[df["Machine failure"] == 1]["Torque"].mean(), 2)
    assert r["unit"] == "Nm"
    r = A.sensor_statistic(df, "rotational_speed", "max")
    assert r["value"] == df["Rotational speed"].max()


def test_derived_power_sensor(df):
    r = A.sensor_statistic(df, "power", "mean")
    assert 5000 < r["value"] < 7000          # around 2860 W * ... typical ~6000 W


def test_threshold(df):
    r = A.count_threshold(df, "tool_wear", ">", 200)
    sub = df[df["Tool wear"] > 200]
    assert r["count"] == len(sub)
    assert r["of_which_failed"] == int(sub["Machine failure"].sum())


def test_compare(df):
    r = A.compare_failed_vs_normal(df, "tool_wear")
    assert r["mean_failed"] > r["mean_normal"]          # worn tools fail more


def test_invalid_arguments_are_rejected(df):
    with pytest.raises(A.AnalyticsError):
        A.sensor_statistic(df, "voltage")
    with pytest.raises(A.AnalyticsError):
        A.count_machines(df, failure_mode="XYZ")
    with pytest.raises(A.AnalyticsError):
        A.count_threshold(df, "torque", "==", 40)
    with pytest.raises(A.AnalyticsError):
        A.sensor_statistic(df, "torque", "variance")


def test_empty_filter_gives_no_value(df):
    tiny = df.head(5)
    r = A.sensor_statistic(tiny, "torque", failure_mode="RNF")
    assert r["records"] == 0 and r["value"] is None
