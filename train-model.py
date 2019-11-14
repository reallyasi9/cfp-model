#!/usr/bin/env python3

import os.path
import pandas as pd
from logbook import Logger, FileHandler, StderrHandler
import click
import io

# from utils import archive
import numpy as np
from keras.utils import Sequence

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
    LOG.debug(f"Number of team-games after dropping short seasons: {df.shape[0]}")

    # Split out the latest year for use as a test set
    df_latest = df.loc[df["Year"] == df["Year"].max()]
    LOG.debug(
        f"Dropping {df_latest.shape[0]} team-games from latest year {df.Year.max()}"
    )
    df = df.drop(df_latest.index, axis="index")
    LOG.debug(f"Number of team-games after dropping most recent year: {df.shape[0]}")

    # Drop games that don't have a final score
    df = df.loc[(~df["Points"].isnull()) & (~df["OPoints"].isnull())]
    LOG.debug(f"Number of team-games after dropping incomplete games: {df.shape[0]}")

    # Drop games before 1958 (modern football only)
    df = df.loc[(df["Year"] >= 1958)]
    LOG.debug(f"Number of team-games after dropping pre-modern seasons: {df.shape[0]}")

    # Convert team to categorical
    df.loc[:, "Team"] = df.loc[:, "Team"].astype("category")
    n_teams = len(df["Team"].cat.categories)
    LOG.debug(f"Number of Teams: {n_teams}")
    # Convert opponent to categorical using the same levels
    df.loc[:, "Opponent"] = df.loc[:, "Opponent"].astype(df["Team"].dtype)
    LOG.debug(
        f"Number of games with Opponent not in Team list: {(df.Opponent.cat.codes == -1).sum()}"
    )
    # Force unknown opponents to a positive number for NN purposes
    df["Opponent"] = df.Opponent.cat.add_categories(["Unknown"])
    df["Opponent"] = df.Opponent.fillna("Unknown")

    # Batch by team and year, sorting by week number
    df = df.sort_values(["Year", "Week", "Team"])
    grouped_df = df.groupby(["Team", "Year"], sort=False)
    LOG.debug(f"Number of team-season batches after drops: {len(grouped_df)}")

    LOG.debug(f"Longest season: {grouped_df.size().max()}")

    # Convert to a list of arrays for NN input
    teams, opponents, weeks, years, homes, wins, margins = zip(
        *[
            (
                g["Team"].cat.codes.values,
                g["Opponent"].cat.codes.values,
                g["Week"].values,
                g["Year"].values,
                g["Home"].values,
                g["Win"].values,
                (g["Points"] - g["OPoints"]).values,
            )
            for _, g in grouped_df
        ]
    )

    # With these, we can begin.
    teams_train, teams_test, opponents_train, opponents_test, weeks_train, weeks_test, years_train, years_test, homes_train, homes_test, wins_train, wins_test, margins_train, margins_test = train_test_split(
        teams,
        opponents,
        weeks,
        years,
        homes,
        wins,
        margins,
        test_size=0.25,
        random_state=0xDEADBEEF,
    )
    LOG.debug(f"Train size {len(teams_train)}, test size {len(teams_test)}")

    team_input = Input(shape=(1,), name="team_input")
    opponent_input = Input(shape=(1,), name="opponent_input")
    nn_embed = Embedding(n_teams + 1, 15)
    nn_team = nn_embed(team_input)
    nn_opponent = nn_embed(opponent_input)
    week_input = Input(shape=(1, 1), name="week_input")
    nn = Concatenate()([nn_team, nn_opponent, week_input])
    nn = LSTM(128, return_sequences=True)(nn)
    nn = Dropout(0.1)(nn)
    nn = LSTM(128, return_sequences=False)(nn)
    nn = Dropout(0.1)(nn)
    prediction = Dense(1, activation="sigmoid", name="win_output")(nn)

    model = Model(inputs=[team_input, opponent_input, week_input], outputs=[prediction])
    ls = LogSummary()
    model.summary(print_fn=ls)
    ls.log("INFO")

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    LOG.info("Fitting model")
    model.fit_generator(
        InputSequence(teams_train, opponents_train, weeks_train, wins_train),
        verbose=1,
        epochs=10,
        use_multiprocessing=True,
        workers=-1,
    )

    LOG.info("Testing model")
    ev = model.evaluate_generator(
        InputSequence(teams_test, opponents_test, weeks_test, wins_test)
    )
    LOG.info(f"Evaluation: {ev}")

    LOG.info("Done!")


class LogSummary:
    def __init__(self):
        self.buf = io.StringIO()

    def __call__(self, s):
        self.buf.write(s + "\n")

    def log(self, level):
        LOG.log(level, "\n" + self.buf.getvalue())


class InputSequence(Sequence):
    def __init__(self, teams, opponents, weeks, wins):
        self.teams = teams
        self.opponents = opponents
        self.wins = wins
        self.weeks = weeks

    def __len__(self):
        return len(self.teams)

    def __getitem__(self, idx):
        return (
            {
                "team_input": self.teams[idx],
                "opponent_input": self.opponents[idx],
                "week_input": np.reshape(self.weeks[idx], (len(self.weeks[idx]), 1, 1)),
            },
            {"win_output": self.wins[idx]},
        )


if __name__ == "__main__":
    with DEFAULT_HANDLER.applicationbound():
        with DEBUG_HANDLER.applicationbound():
            LOG.info("Beginning processing")
            train_model()
