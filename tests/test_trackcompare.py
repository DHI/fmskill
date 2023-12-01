import pytest
import pandas as pd
import numpy as np

import modelskill as ms


@pytest.fixture
def obs_tiny_df():
    time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:02",  # duplicate time (not spatially)
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:03",  # duplicate time (not spatially)
            "2017-10-27 13:00:04",
        ]
    )
    x = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0])
    y = np.array([11.0, 12.0, 12.5, 13.0, 13.5, 14.0])
    val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return pd.DataFrame(data={"x": x, "y": y, "alti": val}, index=time)


@pytest.fixture
def obs_tiny(obs_tiny_df):
    with pytest.warns(UserWarning, match="Removed 2 duplicate timestamps"):
        o = ms.TrackObservation(obs_tiny_df, item="alti", x_item="x", y_item="y")
    return o


@pytest.fixture
def mod_tiny3():
    time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:02",  # duplicate
            "2017-10-27 13:00:03",
        ]
    )
    x = np.array([2.0, 2.5, 3.0])
    y = np.array([12.0, 12.5, 13.0])
    val = np.array([2.1, 3.1, 4.1])
    df = pd.DataFrame(data={"x": x, "y": y, "m1": val}, index=time)
    with pytest.warns(UserWarning, match="Removed 1 duplicate timestamps"):
        mr = ms.TrackModelResult(df, item="m1", x_item="x", y_item="y")
    return mr


@pytest.fixture
def mod_tiny_3last():
    time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:03",  # duplicate time (not spatially)
            "2017-10-27 13:00:04",
        ]
    )
    x = np.array([3.0, 3.5, 4.0])
    y = np.array([13.0, 13.5, 14.0])
    val = np.array([4.1, 5.1, 6.1])
    df = pd.DataFrame(data={"x": x, "y": y, "m1": val}, index=time)
    with pytest.warns(UserWarning, match="Removed 1 duplicate timestamps"):
        mr = ms.TrackModelResult(df, item="m1", x_item="x", y_item="y")
    return mr


@pytest.fixture
def mod_tiny_unique():
    """Model match observation, except for duplicate time (removed)"""
    time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            # "2017-10-27 13:00:02",  # duplicate time (not spatially)
            "2017-10-27 13:00:03",
            # "2017-10-27 13:00:03",  # duplicate time (not spatially)
            "2017-10-27 13:00:04",
        ]
    )
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([11.0, 12.0, 13.0, 14.0])
    val = np.array([1.1, 2.1, 4.1, 6.1])
    df = pd.DataFrame(data={"x": x, "y": y, "m1": val}, index=time)
    return ms.TrackModelResult(df, item="m1", x_item="x", y_item="y")


@pytest.fixture
def mod_tiny_rounding_error():
    """Model match observation, but with rounding error on x,y"""
    time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:02",  # duplicate time (not spatially)
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:03",  # duplicate time (not spatially)
            "2017-10-27 13:00:04",
        ]
    )
    x = np.array([1.0, 2.0, 2.50001, 3.0, 3.50001, 4.0])
    y = np.array([11.0, 12.0, 12.5, 13.0, 13.50001, 14.0])
    val = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
    df = pd.DataFrame(data={"x": x, "y": y, "m1": val}, index=time)
    with pytest.warns(UserWarning, match="duplicate"):
        mr = ms.TrackModelResult(df, item="m1", x_item="x", y_item="y")
    return mr


@pytest.fixture
def mod_tiny_longer():
    """Model match observation, but with more data"""
    time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
            "2017-10-27 13:00:04",
            "2017-10-27 13:00:05",
        ]
    )
    x = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    y = np.array([11.0, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0])
    val = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1])
    df = pd.DataFrame(data={"x": x, "y": y, "m1": val}, index=time)
    # with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
    mr = ms.TrackModelResult(df, item="m1", x_item="x", y_item="y")
    return mr


# TODO: add some with missing values


def test_tiny_obs_offset(obs_tiny_df):
    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        obs_tiny = ms.TrackObservation(
            obs_tiny_df, item="alti", x_item="x", y_item="y", keep_duplicates="offset"
        )
    assert len(obs_tiny) == 6
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02.001",
            "2017-10-27 13:00:02.002",
            "2017-10-27 13:00:03.003",
            "2017-10-27 13:00:03.004",
            "2017-10-27 13:00:04",
        ]
    )
    expected_x = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0])
    expected_y = np.array([11.0, 12.0, 12.5, 13.0, 13.5, 14.0])
    expected_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert obs_tiny.time.equals(expected_time)
    assert np.all(obs_tiny.x == expected_x)
    assert np.all(obs_tiny.y == expected_y)
    assert np.all(obs_tiny.values == expected_val)


def test_tiny_obs_first(obs_tiny_df):
    with pytest.warns(UserWarning, match="Removed 2 duplicate timestamps"):
        obs_tiny = ms.TrackObservation(
            obs_tiny_df, item="alti", x_item="x", y_item="y", keep_duplicates="first"
        )

    assert len(obs_tiny) == 4
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
        ]
    )
    expected_x = np.array([1.0, 2.0, 3.0, 4.0])
    expected_y = np.array([11.0, 12.0, 13.0, 14.0])
    expected_val = np.array([1.0, 2.0, 4.0, 6.0])
    assert obs_tiny.time.equals(expected_time)
    assert np.all(obs_tiny.x == expected_x)
    assert np.all(obs_tiny.y == expected_y)
    assert np.all(obs_tiny.values == expected_val)


def test_tiny_obs_last(obs_tiny_df):
    with pytest.warns(UserWarning, match="Removed 2 duplicate timestamps"):
        obs_tiny = ms.TrackObservation(
            obs_tiny_df, item="alti", x_item="x", y_item="y", keep_duplicates="last"
        )

    assert len(obs_tiny) == 4
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
        ]
    )
    expected_x = np.array([1.0, 2.5, 3.5, 4.0])
    expected_y = np.array([11.0, 12.5, 13.5, 14.0])
    expected_val = np.array([1.0, 3.0, 5.0, 6.0])
    assert obs_tiny.time.equals(expected_time)
    assert np.all(obs_tiny.x == expected_x)
    assert np.all(obs_tiny.y == expected_y)
    assert np.all(obs_tiny.values == expected_val)


def test_tiny_obs_False(obs_tiny_df):
    with pytest.warns(UserWarning, match="Removed 4 duplicate timestamps"):
        obs_tiny = ms.TrackObservation(
            obs_tiny_df, item="alti", x_item="x", y_item="y", keep_duplicates=False
        )

    assert len(obs_tiny) == 2
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:04",
        ]
    )
    expected_x = np.array([1.0, 4.0])
    expected_y = np.array([11.0, 14.0])
    expected_val = np.array([1.0, 6.0])
    assert obs_tiny.time.equals(expected_time)
    assert np.all(obs_tiny.x == expected_x)
    assert np.all(obs_tiny.y == expected_y)
    assert np.all(obs_tiny.values == expected_val)


def test_tiny_mod3(obs_tiny, mod_tiny3):
    cmp = ms.compare(obs_tiny, mod_tiny3)[0]
    assert cmp.n_points == 2
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
        ]
    )
    assert cmp.time.equals(expected_time)
    assert np.all(cmp.x == np.array([2.0, 3.0]))


def test_tiny_mod_3last(obs_tiny, mod_tiny_3last):
    cmp = ms.compare(obs_tiny, mod_tiny_3last)[0]
    assert cmp.n_points == 2
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
        ]
    )
    assert cmp.time.equals(expected_time)
    assert np.all(cmp.x == np.array([3.0, 4.0]))


def test_tiny_mod_unique(obs_tiny, mod_tiny_unique):
    cmp = ms.compare(obs_tiny, mod_tiny_unique)[0]
    assert cmp.n_points == 4
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
        ]
    )
    assert cmp.time.equals(expected_time)
    assert np.all(cmp.x == np.array([1.0, 2.0, 3.0, 4.0]))


# Currently fails as check on x, y difference is missing!
def test_tiny_mod_xy_difference(obs_tiny_df, mod_tiny_unique):
    obs_tiny_df.x.iloc[0] = 1.1  # difference in x larger than tolerance
    obs_tiny_df.y.iloc[3] = 13.6  # difference in y larger than tolerance
    with pytest.warns(UserWarning, match="Removed 2 duplicate timestamps"):
        obs_tiny = ms.TrackObservation(
            obs_tiny_df, item="alti", x_item="x", y_item="y", keep_duplicates="first"
        )
    cmp = ms.compare(obs_tiny, mod_tiny_unique)[0]
    assert cmp.n_points == 2  # 2 points removed due to difference in x,y
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:04",
        ]
    )
    assert cmp.time.equals(expected_time)
    assert np.all(cmp.x == np.array([2.0, 4.0]))


def test_tiny_mod_rounding_error(obs_tiny, mod_tiny_rounding_error):
    # accepts rounding error in x, y
    cmp = ms.compare(obs_tiny, mod_tiny_rounding_error)[0]
    assert cmp.n_points == 4
    expected_time = pd.DatetimeIndex(
        [
            "2017-10-27 13:00:01",
            "2017-10-27 13:00:02",
            "2017-10-27 13:00:03",
            "2017-10-27 13:00:04",
        ]
    )
    assert cmp.time.equals(expected_time)
    assert np.all(cmp.x == np.array([1.0, 2.0, 3.0, 4.0]))


@pytest.fixture
def observation_df():
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    return pd.read_csv(fn, index_col=0, parse_dates=True)


@pytest.fixture
def observation(observation_df):
    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        o = ms.TrackObservation(observation_df, item=2, name="alti")
    return o


@pytest.fixture
def modelresult():
    fn = "tests/testdata/NorthSeaHD_extracted_track.dfs0"
    with pytest.warns(UserWarning, match="Removed 22 duplicate timestamps"):
        mr = ms.ModelResult(fn, gtype="track", item=2, name="HD")
    return mr


@pytest.fixture
def comparer(observation, modelresult):
    return ms.compare(observation, modelresult)


def test_skill(comparer):
    c = comparer
    df = c.skill().df

    assert df.loc["alti"].n == 532  # 544


# def test_extract_no_time_overlap(modelresult, observation_df):
#     mr = modelresult
#     df = observation_df.copy(deep=True)
#     df.index = df.index + np.timedelta64(100, "D")
#     with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
#         o = ms.TrackObservation(df, item=2, name="alti")

#     with pytest.raises(ValueError, match="Validation failed"):
#         with pytest.warns(UserWarning, match="No time overlap!"):
#             ms.Connector(o, mr)

#     with pytest.warns(UserWarning, match="No time overlap!"):
#         con = ms.Connector(o, mr, validate=False)

#     with pytest.warns(UserWarning, match="No overlapping data"):
#         cc = con.extract()

#     assert cc.n_comparers == 0


# def test_extract_obs_start_before(modelresult, observation_df):
#     mr = modelresult
#     df = observation_df.copy(deep=True)
#     df.index = df.index - np.timedelta64(1, "D")
#     with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
#         o = ms.TrackObservation(df, item=2, name="alti")
#     con = ms.Connector(o, mr)
#     with pytest.warns(UserWarning, match="No overlapping data"):
#         cc = con.extract()
#     assert cc.n_comparers == 0


# def test_extract_obs_end_after(modelresult, observation_df):
#     mr = modelresult
#     df = observation_df.copy(deep=True)
#     df.index = df.index + np.timedelta64(1, "D")
#     with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
#         o = ms.TrackObservation(df, item=2, name="alti")
#     con = ms.Connector(o, mr)
#     with pytest.warns(UserWarning, match="No overlapping data"):
#         cc = con.extract()
#     assert cc.n_comparers == 0


# def test_extract_no_spatial_overlap_dfs0(modelresult, observation_df):
#     mr = modelresult
#     df = observation_df.copy(deep=True)
#     df.lon = -100
#     df.lat = -50
#     # with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
#     o = ms.TrackObservation(df, item=2, name="alti")
#     con = ms.Connector(o, mr)
#     with pytest.warns(UserWarning, match="No overlapping data"):
#     cc = con.extract()

# assert cc.n_comparers == 0
# assert cc.n_points == 0


# def test_extract_no_spatial_overlap_dfsu(observation_df):


def test_skill_vs_spatial_skill(comparer):
    df = comparer.skill().df  # to compare to result of .skill()
    ds = comparer.spatial_skill(bins=1)  # force 1 bin only

    assert df.loc["alti"].n == ds.n.values
    assert df.loc["alti"].bias == ds.ds.bias.values
    assert ds.x.size == 1
    assert ds.y.size == 1
    # assert ds.coords._names == {"x","y"}  # TODO: Why return "observation" as by, when n_obs==1 but not "model"?


def test_spatial_skill_bins(comparer):
    # default
    ds = comparer.spatial_skill(metrics=["bias"])
    assert len(ds.x) == 5
    assert len(ds.y) == 5

    # float
    ds = comparer.spatial_skill(metrics=["bias"], bins=2)
    assert len(ds.x) == 2
    assert len(ds.y) == 2

    # float for x and range for y
    ds = comparer.spatial_skill(metrics=["bias"], bins=(2, [50, 50.5, 51, 53]))
    assert len(ds.x) == 2
    assert len(ds.y) == 3

    # binsize (overwrites bins)
    ds = comparer.spatial_skill(metrics=["bias"], binsize=2.5, bins=100)
    assert len(ds.x) == 4
    assert len(ds.y) == 3
    assert ds.x[0] == -0.75


def test_spatial_skill_by(comparer):
    # odd order of by
    ds = comparer.spatial_skill(metrics=["bias"], by=["y", "mod"])
    assert ds.coords._names == {"y", "model", "x"}


def test_spatial_skill_misc(comparer):
    # miniumum n
    ds = comparer.spatial_skill(metrics=["bias", "rmse"], n_min=20)
    df = ds.to_dataframe()
    assert df.loc[df.n < 20, ["bias", "rmse"]].size == 30
    assert df.loc[df.n < 20, ["bias", "rmse"]].isna().all().all()


def test_hist(comparer):
    cc = comparer

    with pytest.warns(FutureWarning):
        cc.hist()

    cc.plot.hist(bins=np.linspace(0, 7, num=15))

    cc[0].plot.hist(bins=10)
    cc[0].plot.hist(density=False)
    cc[0].plot.hist(model=0, title="new_title", alpha=0.2)


def test_residual_hist(comparer):
    cc = comparer
    cc[0].plot.residual_hist()
    cc[0].plot.residual_hist(bins=10, title="new_title", color="blue")
