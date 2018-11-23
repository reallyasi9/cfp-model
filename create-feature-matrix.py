#!/usr/bin/env python3

import os.path
import sys
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
