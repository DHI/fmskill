"""
The `observation` module contains different types of Observation classes for
fixed locations (PointObservation), or locations moving in space (TrackObservation).

Examples
--------
>>> o1 = PointObservation("klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
"""
from dataclasses import dataclass
import os
from typing import List, Optional, Sequence, Union
import numpy as np
import pandas as pd
import mikeio
from copy import deepcopy

from .utils import make_unique_index, _get_name
from .types import Quantity
from .timeseries import TimeSeries


# TODO: remove this
def _parse_item(items, item, item_str="item"):
    if isinstance(item, int):
        item = len(items) + item if (item < 0) else item
        if (item < 0) or (item >= len(items)):
            raise IndexError(f"{item_str} is out of range (0, {len(items)})")
    elif isinstance(item, str):
        item = items.index(item)
    else:
        raise TypeError(f"{item_str} must be int or string")
    return item


@dataclass
class ItemSelection:
    """Utility class to keep track of value, position and auxiliary items"""

    val: str
    x: Optional[str] = None
    y: Optional[str] = None
    z: Optional[str] = None
    aux: Sequence[str]

    def __post_init__(self):
        # check that val, pos and aux are unique, and that they are not overlapping
        all_items = self.all
        if len(all_items) != len(set(all_items)):
            raise ValueError("Items must be unique")

    @property
    def pos(self):
        pos = []
        if self.x is not None:
            pos.append(self.x)
        if self.y is not None:
            pos.append(self.y)
        if self.z is not None:
            pos.append(self.z)
        return pos

    @property
    def all(self) -> Sequence[str]:
        return self.pos + [self.val] + self.aux


def _parse_items(
    items: List[Union[str, int]],
    val_item: Optional[Union[str, int]] = None,
    x_item: Optional[Union[str, int]] = None,
    y_item: Optional[Union[str, int]] = None,
    z_item: Optional[Union[str, int]] = None,
    aux_items: Optional[List[Union[str, int]]] = None,
) -> ItemSelection:
    """Parse items and return val, position and auxiliary items
    Default behaviour:
    - x_item is first item
    - y_item is second item
    - z_item is None
    - val_item is third item (if more than 2 items)
    - aux_items are None

    Both integer and str are accepted as items. If str, it must be a key in data.
    """
    items = list(items)
    min_items = 1 if x_item is None else 3
    assert len(items) >= min_items, f"data must contain at least {min_items} item(s)"
    if val_item is None:
        val_item = items[0]
    else:
        val_item = _get_name(val_item, items)

    # Check existance of items and convert to names
    x_item = _get_name(x_item, items) if x_item is not None else None
    y_item = _get_name(y_item, items) if y_item is not None else None
    z_item = _get_name(z_item, items) if z_item is not None else None
    pos_items = None
    if x_item is not None and y_item is not None:
        # TODO: should we allow z_item if x_item and y_item are None?
        pos_items = [x_item, y_item]
        if z_item is not None:
            pos_items.append(z_item)

    if aux_items is not None:
        aux_items = [_get_name(a, items) for a in aux_items]

    # Check overlap and raise errors if any
    if pos_items is not None and val_item in pos_items:
        raise ValueError(f"item {val_item} should not be in x, y or z")
    if aux_items is not None and val_item in aux_items:
        raise ValueError(f"item {val_item} should not be in aux_items")
    if pos_items is not None and aux_items is not None:
        overlapping_items = set(pos_items) & set(aux_items)
        if overlapping_items:
            raise ValueError(
                f"x, y and z items and aux_items should not have overlapping items. Overlapping items: {overlapping_items}"
            )

    return ItemSelection(val=val_item, x=x_item, y=y_item, z=z_item, aux=aux_items)


class Observation(TimeSeries):
    """Base class for observations

    Parameters
    ----------
    data : pd.DataFrame
    name : str, optional
        user-defined name, e.g. "Station A", by default "Observation"
    quantity : Optional[Quantity], optional
        The quantity of the observation, for validation with model results
    weight : float, optional
        weighting factor, to be used in weighted skill scores, by default 1.0
    color : str, optional
        color of the observation in plots, by default "#d62728"
    """

    def __init__(
        self,
        data: pd.DataFrame,
        name: str = "Observation",
        quantity: Optional[Quantity] = None,
        weight: float = 1.0,
        color: str = "#d62728",
    ):
        if name is None:
            name = "Observation"

        # TODO move this to TimeSeries?
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(
                f"Input must have a datetime index! Provided index was {type(data.index)}"
            )
        time = data.index.round(freq="100us")  # 0.0001s accuracy
        data.index = pd.DatetimeIndex(time, freq="infer")

        self.weight = weight

        super().__init__(name=name, data=data, quantity=quantity, color=color)

    @property
    def values(self) -> np.ndarray:
        "Observed values"
        return self.data.values

    @property
    def n_points(self):
        """Number of observations"""
        return len(self.data)

    def copy(self):
        return deepcopy(self)


class PointObservation(Observation):
    """Class for observations of fixed locations

    Create a PointObservation from a dfs0 file or a pd.DataFrame.

    Parameters
    ----------
    data : (str, pd.DataFrame, pd.Series)
        dfs0 filename or dataframe with the data
    item : (int, str), optional
        index or name of the wanted item, by default None
    x : float, optional
        x-coordinate of the observation point, by default None
    y : float, optional
        y-coordinate of the observation point, by default None
    z : float, optional
        z-coordinate of the observation point, by default None
    name : str, optional
        user-defined name for easy identification in plots etc, by default file basename
    quantity : Quantity, optional
        The quantity of the observation, for validation with model results

    Examples
    --------
    >>> o1 = PointObservation("klagshamn.dfs0", item=0, x=366844, y=6154291, name="Klagshamn")
    >>> o1 = PointObservation("klagshamn.dfs0", item="Water Level", x=366844, y=6154291)
    >>> o1 = PointObservation(df, item=0, x=366844, y=6154291, name="Klagshamn")
    >>> o1 = PointObservation(df["Water Level"], x=366844, y=6154291)
    """

    @property
    def geometry(self):
        from shapely.geometry import Point

        if self.z is None:
            return Point(self.x, self.y)
        else:
            return Point(self.x, self.y, self.z)

    def __init__(
        self,
        data,
        *,
        item=None,
        x: float = None,
        y: float = None,
        z: float = None,
        name: str = None,
        quantity: Optional[Union[str, Quantity]] = None,
    ):
        self.x = x
        self.y = y
        self.z = z

        self._filename = None
        self._item = None

        # TODO move this to TimeSeries?
        if isinstance(data, pd.Series):
            df = data.to_frame()
            if name is None:
                name = "Observation"
        elif isinstance(data, mikeio.DataArray):
            df = data.to_dataframe()
            if quantity is None:
                quantity = Quantity.from_mikeio_iteminfo(data.item)
        elif isinstance(data, mikeio.Dataset):
            df = data.to_dataframe()[[item]]
            if quantity is None:
                quantity = Quantity.from_mikeio_iteminfo(data[item].item)
        elif isinstance(data, pd.DataFrame):
            df = data
            default_name = "Observation"
            if item is None:
                if len(df.columns) == 1:
                    item = 0
                else:
                    raise ValueError(
                        f"item must be specified (more than one column in dataframe). Available columns: {list(df.columns)}"
                    )
            self._item = item

            if isinstance(item, str):
                df = df[[item]]
                default_name = item
            elif isinstance(item, int):
                if item < 0:
                    item = len(df.columns) + item
                default_name = df.columns[item]
                df = df.iloc[:, item]
            else:
                raise TypeError("item must be int or string")
            if name is None:
                name = default_name
        elif isinstance(data, str):
            assert os.path.exists(data)
            self._filename = data
            if name is None:
                name = os.path.basename(data).split(".")[0]

            ext = os.path.splitext(data)[-1]
            if ext == ".dfs0":
                df, iteminfo = self._read_dfs0(mikeio.open(data), item)
                if quantity is None:
                    quantity = Quantity.from_mikeio_iteminfo(iteminfo)
            else:
                raise NotImplementedError("Only dfs0 files supported")
        else:
            raise TypeError(
                f"input must be str, mikeio.DataArray/Dataset or pandas Series/DataFrame! Given input has type {type(data)}"
            )

        if not df.index.is_unique:
            # TODO: duplicates_keep="mean","first","last"
            raise ValueError(
                "Time axis has duplicate entries. It must be monotonically increasing."
            )

        super().__init__(
            name=name,
            data=df,
            quantity=quantity,
        )

    def __repr__(self):
        out = f"PointObservation: {self.name}, x={self.x}, y={self.y}"
        return out

    # TODO does this belong here?
    @staticmethod
    def _read_dfs0(dfs, item):
        """Read data from dfs0 file"""
        if item is None:
            if len(dfs.items) == 1:
                item = 0
            else:
                item_names = [i.name for i in dfs.items]
                raise ValueError(
                    f"item needs to be specified (more than one in file). Available items: {item_names} "
                )
        ds = dfs.read(items=item)
        itemInfo = ds.items[0]
        df = ds.to_dataframe()
        df.dropna(inplace=True)
        return df, itemInfo


def _data_to_xarray(
    data: pd.DataFrame,
    val_item=None,
    pos_items=None,
    aux_items=None,
    name=None,
    x=None,
    y=None,
    z=None,
):
    """Convert data to internal xarray.Dataset format"""
    if isinstance(data, pd.DataFrame):
        cols = data.columns
        items = _parse_items(cols, val_item, pos_items, aux_items)
        data = data[items.all]
        data.index.name = "time"
        data.rename(columns={items.obs: "Observation"}, inplace=True)
        data = data.to_xarray()
    else:
        raise ValueError(f"Unknown data type '{type(data)}' (pd.DataFrame)")

    data.attrs["name"] = name if name is not None else items.obs
    data["Observation"].attrs["kind"] = "observation"
    for m in items.model:
        data[m].attrs["kind"] = "model"
    for a in items.aux:
        data[a].attrs["kind"] = "auxiliary"

    if x is not None:
        data["x"] = x
        data["x"].attrs["kind"] = "position"
    if y is not None:
        data["y"] = y
        data["y"].attrs["kind"] = "position"
    if z is not None:
        data["z"] = z
        data["z"].attrs["kind"] = "position"

    return data


class TrackObservation(Observation):
    """Class for observation with locations moving in space, e.g. satellite altimetry

    The data needs in addition to the datetime of each single observation point also, x and y coordinates.

    Create TrackObservation from dfs0 or DataFrame

    Parameters
    ----------
    data : (str, pd.DataFrame)
        path to dfs0 file or DataFrame with track data
    item : (str, int), optional
        item name or index of values, by default 2
    name : str, optional
        user-defined name for easy identification in plots etc, by default file basename
    x_item : (str, int), optional
        item name or index of x-coordinate, by default 0
    y_item : (str, int), optional
        item name or index of y-coordinate, by default 1
    offset_duplicates : float, optional
        in case of duplicate timestamps, add this many seconds to consecutive duplicate entries, by default 0.001


    Examples
    --------
    >>> o1 = TrackObservation("track.dfs0", item=2, name="c2")

    >>> o1 = TrackObservation("track.dfs0", item="wind_speed", name="c2")

    >>> o1 = TrackObservation("lon_after_lat.dfs0", item="wl", x_item=1, y_item=0)

    >>> o1 = TrackObservation("track_wl.dfs0", item="wl", x_item="lon", y_item="lat")

    >>> df = pd.DataFrame(
    ...         {
    ...             "t": pd.date_range("2010-01-01", freq="10s", periods=n),
    ...             "x": np.linspace(0, 10, n),
    ...             "y": np.linspace(45000, 45100, n),
    ...             "swh": [0.1, 0.3, 0.4, 0.5, 0.3],
    ...         }
    ... )
    >>> df = df.set_index("t")
    >>> df
                        x        y  swh
    t
    2010-01-01 00:00:00   0.0  45000.0  0.1
    2010-01-01 00:00:10   2.5  45025.0  0.3
    2010-01-01 00:00:20   5.0  45050.0  0.4
    2010-01-01 00:00:30   7.5  45075.0  0.5
    2010-01-01 00:00:40  10.0  45100.0  0.3
    >>> t1 = TrackObservation(df, name="fake")
    >>> t1.n_points
    5
    >>> t1.values
    array([0.1, 0.3, 0.4, 0.5, 0.3])
    >>> t1.time
    DatetimeIndex(['2010-01-01 00:00:00', '2010-01-01 00:00:10',
               '2010-01-01 00:00:20', '2010-01-01 00:00:30',
               '2010-01-01 00:00:40'],
              dtype='datetime64[ns]', name='t', freq=None)
    >>> t1.x
    array([ 0. ,  2.5,  5. ,  7.5, 10. ])
    >>> t1.y
    array([45000., 45025., 45050., 45075., 45100.])

    """

    @property
    def geometry(self):
        from shapely.geometry import MultiPoint

        """Coordinates of observation"""
        return MultiPoint(self.data.iloc[:, 0:2].values)

    @property
    def x(self):
        return self.data.iloc[:, 0].values

    @property
    def y(self):
        return self.data.iloc[:, 1].values

    @property
    def values(self):
        return self.data.iloc[:, 2].values

    def __init__(
        self,
        data,
        *,
        item: int = None,
        name: str = None,
        x_item=0,
        y_item=1,
        offset_duplicates: float = 0.001,
        quantity: Optional[Quantity] = None,
    ):
        self._filename = None
        self._item = None

        if isinstance(data, pd.DataFrame):
            df = data
            df_items = df.columns.to_list()
            items = self._parse_track_items(df_items, x_item, y_item, item)
            df = df.iloc[:, items].copy()
        elif isinstance(data, str):
            assert os.path.exists(data)
            self._filename = data
            if name is None:
                name = os.path.basename(data).split(".")[0]

            ext = os.path.splitext(data)[-1]
            if ext == ".dfs0":
                dfs = mikeio.open(data)
                file_items = [i.name for i in dfs.items]
                items = self._parse_track_items(file_items, x_item, y_item, item)
                df, iteminfo = self._read_dfs0(dfs, items)
                if quantity is None:
                    quantity = Quantity.from_mikeio_iteminfo(iteminfo)
            else:
                raise NotImplementedError(
                    "Only dfs0 files and DataFrames are supported"
                )
        else:
            raise TypeError(
                f"input must be str or pandas DataFrame! Given input has type {type(data)}"
            )

        # A unique index makes lookup much faster O(1)
        if not df.index.is_unique:
            df.index = make_unique_index(df.index, offset_duplicates=offset_duplicates)

        # TODO is this needed elsewhere?
        # make sure location columns are named x and y
        if isinstance(x_item, str):
            old_x_name = x_item
        else:
            old_x_name = df.columns[x_item]

        if isinstance(y_item, str):
            old_y_name = y_item
        else:
            old_y_name = df.columns[y_item]

        df = df.rename(columns={old_x_name: "x", old_y_name: "y"})

        super().__init__(
            name=name,
            data=df,
            quantity=quantity,
        )

    @staticmethod
    def _parse_track_items(items, x_item, y_item, item):
        """If input has exactly 3 items we accept item=None"""
        if len(items) < 3:
            raise ValueError(
                f"Input has only {len(items)} items. It should have at least 3."
            )
        if item is None:
            if len(items) == 3:
                item = 2
            elif len(items) > 3:
                raise ValueError("Input has more than 3 items, but item was not given!")
        else:
            item = _parse_item(items, item)

        x_item = _parse_item(items, x_item, "x_item")
        y_item = _parse_item(items, y_item, "y_item")

        if (item == x_item) or (item == y_item) or (x_item == y_item):
            raise ValueError(
                f"x-item ({x_item}), y-item ({y_item}) and value-item ({item}) must be different!"
            )
        return [x_item, y_item, item]

    def __repr__(self):
        out = f"TrackObservation: {self.name}, n={self.n_points}"
        return out

    @staticmethod
    def _read_dfs0(dfs, items):
        """Read track data from dfs0 file"""
        df = dfs.read(items=items).to_dataframe()
        df.dropna(inplace=True)
        return df, dfs.items[items[-1]]


def unit_display_name(name: str) -> str:
    """Display name

    Examples
    --------
    >>> unit_display_name("meter")
    m
    """

    res = (
        name.replace("meter", "m")
        .replace("_per_", "/")
        .replace(" per ", "/")
        .replace("second", "s")
        .replace("sec", "s")
        .replace("degree", "°")
    )

    return res
