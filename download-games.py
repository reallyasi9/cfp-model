#!/usr/bin/env python3

import requests
import click
import os.path
from bs4 import BeautifulSoup
import re
import pandas as pd
import urllib.parse

YEARS_URL = 'https://www.sports-reference.com/cfb/years/'
YEAR = '2017'

def download_games():
    # Get all years page
    result = requests.get(YEARS_URL)
    result.raise_for_status()

    # Parse out years
    soup = BeautifulSoup(result.content, features='lxml')
    year_nodes = soup.select('table#years th[data-stat="year_id"] a')
    years = {n.string: n['href'] for n in year_nodes}

    # Get specified year page
    year_url = urllib.parse.urljoin(YEARS_URL, years[YEAR])

    result = requests.get(year_url)
    result.raise_for_status()

    # Parse out games nav
    soup = BeautifulSoup(result.content, features='lxml')
    nav_nodes = soup.select('div#inner_nav a')
    nav_node = None
    for n in nav_nodes:
        if n.string == "Schedule & Scores":
            nav_node = n
            break

    # Get games page
    games_url = urllib.parse.urljoin(YEARS_URL, nav_node['href'])

    result = requests.get(games_url)
    result.raise_for_status()

    # Parse out games
    soup = BeautifulSoup(result.content, features='lxml')
    table = soup.select_one('table#schedule')

    df = pd.read_html(str(table))
    df = df[0]  # This returns a list, which is strange, but makes some amount of sense.
    print(df.head(10))


if __name__ == '__main__':
    download_games()
