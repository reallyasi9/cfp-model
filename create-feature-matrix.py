#!/usr/bin/env python3

import os.path
import numpy as np
import pandas as pd
from logbook import Logger, FileHandler, StderrHandler
import click
from utils import archive

DEBUG_HANDLER = FileHandler("create-feature-matrix.log", level="DEBUG", bubble=True)
DEFAULT_HANDLER = StderrHandler(level="INFO")
LOG = Logger("create-feature-matrix")


@click.command()
@click.argument("infile", type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option(
    "--outdir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory",
    default=".",
)
def create_feature_matrix(infile=None, outdir="."):
    # Make output directory
    os.makedirs(outdir, exist_ok=True)

    LOG.info(f"Reading input file {infile}")
    df = pd.read_parquet(infile)

    # List out by team
    out_df = pd.DataFrame()
    out_df = df.loc[
        :,
        [
            "Home",
            "Away",
            "HPoints",
            "APoints",
            "HRanking",
            "ARanking",
            "Week",
            "Year",
            "DateTime",
            "Time",
        ],
    ].rename(
        columns={
            "Home": "Team",
            "Away": "Opponent",
            "HPoints": "Points",
            "APoints": "OPoints",
            "HRanking": "Ranking",
            "ARanking": "ORanking",
        }
    )
    out_df["Home"] = 1
    out_df["Win"] = (out_df["Points"] > out_df["OPoints"]) * 1
    out_df["DOY"] = out_df["DateTime"].map(lambda x: x.timetuple().tm_yday)

    # Create a lagged number of wins (2 seasons' worth sounds good, right?)
    # plus an exponentially-weighted number of wins (older wins are not as
    # predictive, right?)
    out_df.sort_values(by=["Team", "DateTime"], ascending=True, inplace=True)
    roll_win_pct = (
        out_df.groupby("Team").rolling(window="365d", on="DateTime")["Win"].mean()
    )

    def exp_win(s, halflife=np.timedelta64(365, "D")):
        m = np.max(s.index.values)
        x = m - s.index.values
        w = np.exp(-x / (2 * halflife))
        a = np.average(s.values, weights=w)
        return a

    exp_win_pct = (
        out_df.groupby("Team")
        .rolling(window="365d", on="DateTime")["Win"]
        .apply(exp_win, raw=False)
    )

    def lin_win(s, halflife=np.timedelta64(365, "D")):
        m = np.max(s.index.values)
        w = 1.0 - (m - s.index.values) / (2 * halflife)
        a = np.average(s.values, weights=w)
        return a

    lin_win_pct = (
        out_df.groupby("Team")
        .rolling(window="365d", on="DateTime")["Win"]
        .apply(lin_win, raw=False)
    )

    # Join back onto the output by team/datetime,
    # once for the home team and once for the opponent
    out_df = out_df.merge(
        roll_win_pct.to_frame("RollWinPct"),
        how="left",
        left_on=["Team", "DateTime"],
        right_index=True,
    )
    print(roll_win_pct.to_frame("ORollWinPct").rename_axis(["Opponent", "DateTime"]))
    out_df = out_df.merge(
        roll_win_pct.to_frame("ORollWinPct").rename_axis(["Opponent", "DateTime"]),
        how="left",
        left_on=["Opponent", "DateTime"],
        right_index=True,
    )
    print(out_df)
    exit()

    # Join an identical version onto the output but with the teams swapped
    swapped_df = out_df.copy()
    swapped_df.rename(
        columns={
            "Opponent": "Team",
            "Team": "Opponent",
            "OPoints": "Points",
            "Points": "OPoints",
            "ORanking": "Ranking",
            "Ranking": "ORanking",
        },
        inplace=True,
    )
    swapped_df["Home"] = 0
    swapped_df["Win"] = 1 - swapped_df["Win"]

    # Join these to make the final output
    out_df = pd.concat([out_df, swapped_df], sort=True, ignore_index=True)
    LOG.debug(f"\n{out_df}")

    archive(out_df, "feature_matrix", outdir, types=["parquet", "csv"])

    LOG.info("Done!")


if __name__ == "__main__":
    with DEFAULT_HANDLER.applicationbound():
        with DEBUG_HANDLER.applicationbound():
            LOG.info("Beginning processing")
            create_feature_matrix()
