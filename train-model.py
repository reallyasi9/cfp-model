#!/usr/bin/env python3

import os.path
import pandas as pd
from logbook import Logger, FileHandler, StderrHandler
import click
from utils import archive
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, CuDNNLSTM, Embedding, Concatenate

from sklearn.model_selection import train_test_split

DEBUG_HANDLER = FileHandler("train-model.log", level="DEBUG", bubble=True)
DEFAULT_HANDLER = StderrHandler(level="INFO")
LOG = Logger("train-model")


@click.command()
@click.argument("infile", type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option(
    "--outdir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory",
    default=".",
)
def train_model(infile=None, outdir="."):
    # Make output directory
    os.makedirs(outdir, exist_ok=True)

    LOG.info(f"Reading input file {infile}")
    df = pd.read_parquet(infile)
    LOG.debug(f"Read {df.shape[0]} team-games")

    # Batch by team and season and drop smaller seasons.
    # This should drop both schools that are not Division 1 and ancient years
    # that had way fewer games per season and much different rules than today's
    # game.
    grouped_df = df.groupby(["Team", "Year"], sort=False)
    LOG.debug(f"Number of team-season batches: {len(grouped_df)}")

    short_season = grouped_df["Team"].transform("size")
    short_season = short_season < 8
    LOG.debug(f"Number of team-games from short seasons: {short_season.sum()}")

    df = df.loc[~short_season]
    LOG.debug(
        f"Number of team-games after dropping short seasons: {df.shape[0]}")

    # Split out the latest year for use as a test set
    df_latest = df.loc[df["Year"] == df["Year"].max()]
    LOG.debug(
        f"Dropping {df_latest.shape[0]} team-games from latest year {df.Year.max()}")
    df.drop(df_latest.index, axis="index", inplace=True)
    LOG.debug(
        f"Number of team-games after dropping most recent year: {df.shape[0]}")

    # Drop games that don't have a final score
    df = df.loc[(~df["Points"].isnull()) & (~df["OPoints"].isnull())]
    LOG.debug(
        f"Number of team-games after dropping incomplete games: {df.shape[0]}")

    # Convert team to categorical
    df["Team"] = df["Team"].astype("category")
    n_teams = len(df["Team"].cat.categories)
    LOG.debug(f"Number of Teams: {n_teams}")
    # Convert opponent to categorical using the same levels
    df["Opponent"] = df["Opponent"].astype(df["Team"].dtype)
    LOG.debug(
        f"Number of games with Opponent not in Team list: {(df.Opponent.cat.codes == -1).sum()}")

    # Batch by team and year, sorting by week number
    df.sort_values(["Year", "Week", "Team"], inplace=True)
    grouped_df = df.groupby(["Team", "Year"], sort=False)
    LOG.debug(f"Number of team-season batches after drops: {len(grouped_df)}")

    # Convert to a list of arrays for NN input
    teams, opponents, weeks, years, homes, wins, margins = zip(*[(g["Team"].cat.codes, g["Opponent"].cat.codes, g["Week"], g["Year"],
                                                                  g["Home"], g["Win"], g["Points"] - g["OPoints"]) for _, g in grouped_df])

    # With these, we can begin.
    teams_train, teams_test, opponents_train, opponents_test, \
        weeks_train, weeks_test, years_train, years_test, \
        homes_train, homes_test, wins_train, wins_test, \
        margins_train, margins_test = train_test_split(
            teams, opponents, weeks, years, homes, wins, margins, test_size=.25, random_state=0xdeadbeef)
    LOG.debug(f"Train size {len(teams_train)}, test size {len(teams_test)}")

    team_input = Input(shape=(1,))
    nn_team = Embedding(n_teams+1, 10)(team_input)
    opponent_input = Input(shape=(1,))
    nn_opponent = Embedding(n_teams+1, 10)(opponent_input)
    nn = Concatenate()([nn_team, nn_opponent])
    nn = LSTM(64, return_sequences=True)(nn)
    nn = Dropout(0.1)(nn)
    nn = LSTM(64, return_sequences=False)(nn)
    nn = Dropout(0.1)(nn)
    nn = Dense(1, activation="sigmoid")(nn)

    model = Model(inputs=[team_input, opponent_input], outputs=nn)
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit([teams_train, opponents_train], [wins], batch_size=1, verbose=1, validation_split=0.1)

    LOG.info("Done!")


if __name__ == "__main__":
    with DEFAULT_HANDLER.applicationbound():
        with DEBUG_HANDLER.applicationbound():
            LOG.info("Beginning processing")
            train_model()
