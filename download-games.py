#!/usr/bin/env python3

import requests
import os.path
import sys
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import urllib.parse
import hashlib
from logbook import Logger, FileHandler, StderrHandler
import click


DEBUG_HANDLER = FileHandler("download-games.log", level="DEBUG", bubble=True)
DEFAULT_HANDLER = StderrHandler(level="INFO")
LOG = Logger("download-games")
YEAR = "2018"


def archive(df, name, path, types=["feather"]):
    """
    Convenience function for writing dataframe to a given location for
    later use, only archiving if the file doesn't exist.
    """
    h = hashlib.sha256(str(df).encode()).hexdigest()[-8:]
    bn = os.path.basename(name)
    bn = os.path.splitext(bn)[0]
    for t in types:
        if t == "feather":
            ofn = os.path.join(path, f"{bn}-{h}.feather")
            if not os.path.exists(ofn):
                LOG.info(f'Archiving DataFrame to "{ofn}"')
                df.reset_index().to_feather(ofn)
            else:
                LOG.info(f'Archived DataFrame "{ofn}" already exists: skipping')
        elif t == "parquet":
            ofn = os.path.join(path, f"{bn}-{h}.parquet")
            if not os.path.exists(ofn):
                LOG.info(f'Archiving DataFrame to "{ofn}"')
                df.reset_index().to_parquet(ofn)
            else:
                LOG.info(f'Archived DataFrame "{ofn}" already exists: skipping')
        elif t == "csv":
            ofn = os.path.join(path, f"{bn}-{h}.csv")
            if not os.path.exists(ofn):
                LOG.info(f'Archiving DataFrame to "{ofn}"')
                df.reset_index().to_csv(ofn)
            else:
                LOG.info(f'Archived DataFrame "{ofn}" already exists: skipping')
        else:
            raise ValueError(f'Archive type "{t}" not understood')


@click.command()
@click.option(
    "--url",
    "-u",
    "years_url",
    help="URL of yearly CFB game archive list",
    default="https://www.sports-reference.com/cfb/years/",
    show_default=True,
)
@click.option(
    "--year",
    "-y",
    "years",
    multiple=True,
    type=int,
    help="Years to download  [default: None (downloads all years found)]",
)
@click.option(
    "--outdir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory",
    default=".",
)
def download_games(
    years_url="https://www.sports-reference.com/cfb/years/", years=[], outdir="."
):
    # Make output directory
    os.makedirs(outdir, exist_ok=True)

    # Get all years page
    year_urls = download_years(years_url)

    def year_itr(year_urls):
        for year, url in year_urls.items():
            if len(years) > 0 and year not in years:
                continue
            games_url = scrape_games_url(urllib.parse.urljoin(years_url, url))
            df = download_game(urllib.parse.urljoin(years_url, games_url))
            df["Year"] = year
            yield clean_games(df)

    # Concat all years
    df = pd.concat(year_itr(year_urls), sort=False, ignore_index=True)
    LOG.debug(f"\n{df.head(10)}")

    # Save
    min_year = df["Year"].min()
    max_year = df["Year"].max()
    archive(
        df, f"all_games_{min_year}_to_{max_year}", path=".", types=["parquet", "csv"]
    )

    LOG.info("Done!")


def soup_request(url):
    """ Download and soupify the content at `url` """
    result = requests.get(url)
    result.raise_for_status()
    return BeautifulSoup(result.content, features="lxml")


def download_years(url):
    LOG.info(f"Downloading years from {url}")
    soup = soup_request(url)

    # Parse out years
    year_nodes = soup.select('table#years th[data-stat="year_id"] a')
    return {int(n.string): n["href"] for n in year_nodes}


def scrape_games_url(url):
    LOG.info(f"Scraping games url from {url}")
    soup = soup_request(url)

    # Parse out games nav
    nav_nodes = soup.select("div#inner_nav a")
    nav_node = None
    for n in nav_nodes:
        if n.string == "Schedule & Scores":
            nav_node = n
            break

    # Get games page
    return nav_node["href"]


def download_game(url):
    LOG.info(f"Downloading game table from {url}")
    soup = soup_request(url)
    table = soup.select_one("table#schedule")

    df = pd.read_html(str(table))
    df = df[0]  # This returns a list, which is strange, but makes some amount of sense.

    # There are some not-useful rows here, so skip them
    return df.loc[df.Rk != "Rk"]


def clean_games(df):
    LOG.info(f"Cleaning {df.shape[0]} games")
    # Convert columns
    df["Week"] = df["Wk"].astype(int)
    df["WPoints"] = df["Pts"].astype(float)  # some points could be empty, so NAs needed
    df["LPoints"] = df["Pts.1"].astype(float)

    # Some tables don't have a TIME column, so deal with that...
    at_col = 7
    if "Time" not in df.columns:
        LOG.debug(f"No 'Time' column: making one up")
        df["Time"] = "00:00:00"
        at_col -= 1
        LOG.debug(f"'Where' column now {at_col} ('{df.columns[at_col]}')")
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"].fillna("00:00:00"))

    # Extract rankings
    df["WRanking"] = df["Winner"].str.extract(r"^\((\d+)\)", expand=False)
    df["Winner"] = df["Winner"].str.replace(r"^\(\d+\)\s*", "")
    df["LRanking"] = df["Loser"].str.extract(r"^\((\d+)\)", expand=False)
    df["Loser"] = df["Loser"].str.replace(r"^\(\d+\)\s*", "")

    # Turn winners/losers into home/away (ignore neutral sites for now)
    df["Home"] = df["Winner"]
    df["HPoints"] = df["WPoints"]
    df["HRanking"] = df["WRanking"]
    df["Away"] = df["Loser"]
    df["APoints"] = df["LPoints"]
    df["ARanking"] = df["LRanking"]
    df["Where"] = df[df.columns[at_col]].astype(str)
    swap = df["Where"] == "@"
    LOG.debug(f"Need to swap {swap.sum()} Home/Away teams")

    df.loc[swap, "Home"] = df.loc[swap, "Loser"]
    df.loc[swap, "HPoints"] = df.loc[swap, "LPoints"]
    df.loc[swap, "HRanking"] = df.loc[swap, "LRanking"]
    df.loc[swap, "Away"] = df.loc[swap, "Winner"]
    df.loc[swap, "APoints"] = df.loc[swap, "WPoints"]
    df.loc[swap, "ARanking"] = df.loc[swap, "WRanking"]

    df["Spread"] = df["WPoints"] - df["LPoints"]
    df["OU"] = df["WPoints"] + df["LPoints"]

    LOG.debug(
        f"Dropping column {at_col} ('{df.columns[at_col]}') from {df.shape[1]} columns"
    )
    df.drop(df.columns[at_col], axis="columns", inplace=True)
    drop_me = ["Rk", "Wk", "Pts", "Pts.1", "Where"]
    LOG.debug(f"Dropping processed columns {drop_me} from {df.shape[1]} columns")
    df.drop(drop_me, axis="columns", inplace=True)
    LOG.debug(f"Left with DataFrame of shape {df.shape}")
    return df


if __name__ == "__main__":
    with DEFAULT_HANDLER.applicationbound():
        with DEBUG_HANDLER.applicationbound():
            LOG.info("Beginning processing")
            download_games()
