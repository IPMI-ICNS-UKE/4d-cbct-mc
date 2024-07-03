"""Basic reading functions for respiratory signals acquired during 4D RT.

(i)4DCT: read_vxp
4D CBCT: read_cbct
Dose delivery: read_linac
"""
import logging
import re
from datetime import date, datetime
from typing import Any, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from ipmi.fused_types import PathLike

logger = logging.getLogger(__name__)


def read_cbct(
    filepath: PathLike, return_date=False
) -> Union[Union[tuple[Union[DataFrame, Any], date], DataFrame], Any]:
    df_raw = pd.read_csv(
        filepath,
        delimiter=",",
        skiprows=0,
        skipfooter=0,
        engine="python",
        names=["amplitude"],
    )
    assert (
        list(df_raw.amplitude.str.find("Date").values).count(0) == 1
    ), f"{filepath} does not contain Date which is mandatory"
    timestamp_str = (
        df_raw[df_raw.amplitude.str.startswith("Date", na=False)]
        .amplitude.values[0]
        .split("=")[-1]
        .split("=")[-1]
    )
    date = datetime.strptime(timestamp_str.strip(), "%Y_%m_%d").date()
    assert (
        list(df_raw.amplitude.str.find("[Data]").values).count(0) == 1
    ), f"{filepath} does not contain [Data] which is necessary"

    skiprows = df_raw.loc[df_raw.amplitude == "[Data]"].index.values[0] + 1
    assert 7 < skiprows < 12

    df = df_raw[skiprows:]
    df.reset_index(drop=True, inplace=True)

    # add time component (samples per second = 15)
    samples_per_second = 15
    dt = 1 / samples_per_second
    time = np.arange(0, len(df.amplitude) * dt, dt)
    assert len(df.amplitude) == len(time)
    df = pd.concat([pd.Series(time, name="time"), df], axis=1)
    df["beam_on"] = 0
    df = df.astype({"time": float, "amplitude": float, "beam_on": float})
    if return_date:
        return df, date
    return df


def read_linac(
    filepath: PathLike, return_infos=False
) -> Union[Union[tuple[DataFrame, tuple[float, float], date], DataFrame], Any]:
    with open(filepath) as file:
        lines = file.readlines()

    begin_left_right, begin_ant_pos, begin_head_feet = [
        i for i, line in enumerate(lines) if line.startswith(r"Zeit [s]	Amplitude [cm]")
    ]
    header = lines[:begin_left_right]

    str_header = "\n".join(header)
    raw_date = re.search(
        r"Gestartet: (\d{2}\.\d{2}\.\d{4}, \d{2}:\d{2}:\d{2})", str_header
    )
    date = datetime.strptime(raw_date.group(1), "%d.%m.%Y, %H:%M:%S")
    gating_limit_lower = re.search(
        r"Unterer Grenzwerte:\s([+-]?([0-9]*[.,])?[0-9]+)", str_header
    )
    gating_limit_upper = re.search(
        r"Oberer Grenzwerte:\s([+-]?([0-9]*[.,])?[0-9]+)", str_header
    )
    gating_limits = (
        float(gating_limit_lower.group(1).replace(",", ".")),
        float(gating_limit_upper.group(1).replace(",", ".")),
    )

    df = pd.read_csv(
        filepath,
        sep="\t",
        skiprows=begin_ant_pos + 1,
        nrows=begin_head_feet - begin_ant_pos - 5,
        names=["time", "amplitude"],
    )
    try:
        df = df.astype({"time": float, "amplitude": float})
    except ValueError:
        df["time"] = df["time"].str.replace(",", ".").astype(float)
        df["amplitude"] = df["amplitude"].str.replace(",", ".").astype(float)
    gating_start_index = next(
        i
        for i, line in enumerate(lines)
        if line.startswith(r"Zeit [s]	1 = Aktivieren, 0 = Deaktivieren")
    )
    n_gating_rows = next(
        i
        for i, line in enumerate(lines[gating_start_index + 1 :])
        if line.startswith("\n")
    )
    gating_df = pd.read_csv(
        filepath,
        sep="\t",
        skiprows=gating_start_index + 1,
        nrows=n_gating_rows,
        names=["time", "beam_on"],
    )
    try:
        gating_df = gating_df.astype({"time": float, "beam_on": float})
    except ValueError:
        gating_df["time"] = gating_df["time"].str.replace(",", ".")
        gating_df = gating_df.astype({"time": float, "beam_on": float})
    master_df = pd.merge_asof(df, gating_df, on="time", direction="forward")
    master_df.beam_on.fillna(1, inplace=True)
    if return_infos:
        return master_df, gating_limits, date
    return df


def read_vxp(filepath: PathLike, return_date=False):
    with open(filepath) as file:
        lines = file.readlines()

    str_lines = "\n".join(lines)
    if return_date:
        raw_date = re.search(r"Date=(\d{1,2}\-\d{1,2}\-\d{4})", str_lines)
        try:
            date = datetime.strptime(raw_date.group(1), "%m-%d-%Y").date()
        except AttributeError:
            logger.warning(f"Could not get date for {filepath}")
            date = datetime.min.date()
    skiprows = next(i for i, line in enumerate(lines) if line.startswith("[Data]")) + 1

    assert 7 < skiprows < 12

    df = pd.read_csv(
        filepath,
        delimiter=",",
        skiprows=skiprows,
        skipfooter=1,
        engine="python",
        names=["amplitude", "phase", "time", "validflag", "ttlin", "mark", "ttlout"],
    )

    df.reset_index(drop=True, inplace=True)
    df = df.astype({"time": float, "amplitude": float, "ttlin": float})
    df.rename(columns={"ttlin": "beam_on"}, inplace=True)
    df.time = df.time - df.time.min()
    # convert ms to s
    df.time = df.time * 10 ** (-3)

    if df.time.max() == df.time.min():
        logger.warning(
            f"{filepath} does not contain a valid timestamp!"
            f"Generate new time component."
        )
        raw_sps = re.search(r"Samples_per_second=(\d{1,3})", str_lines)
        samples_per_second = raw_sps.group(1)
        df.time = pd.Series([(1 / samples_per_second) * x for x in range(len(df.time))])

    # change amplitude sign due to marker block data acquitsion
    df.amplitude = df.amplitude * (-1)
    if return_date:
        return df, date
    return df
