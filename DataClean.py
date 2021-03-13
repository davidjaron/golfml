import numpy as np
import pandas as pd
import sklearn as sk
from itertools import compress, product
from itertools import combinations

KEY_STATS = ["player", "season", "sg_putt", "made_cut", "sg_arg","sg_app","sg_ott","sg_t2g","sg_total"]
KEY_STATS_HISTORICAL = ["player", "tournament name", "season", "sg_putt", "DK_Total_FP", "made_cut", "sg_arg","sg_app","sg_ott","sg_t2g","sg_total"]


def cleanCurrentData(names):
    data = pd.read_csv("PGAData.csv")
    data = data.sample(frac=1)

    data = data[data.columns.intersection(KEY_STATS)]
    data = data[data["player"].isin(names)]
    data = data[data["season"].isin(["2021"])]
    data = data.dropna()
    data = data.drop(["season"], axis=1)
    data = data.set_index(['player']) 
    data = data.groupby(level=0).mean()
    data = data.reset_index()

    data.to_csv('current_season_stats.csv', index=False, header=False)

def currentPlayerNames():
    data = pd.read_csv("DKSalaries.csv")
    return data["Name"].unique()

def getSeason(year):
    data = pd.read_csv("PGAData.csv")
    data = data[data.columns.intersection(KEY_STATS_HISTORICAL)]
    data = data[data["season"].isin([year])]

    new = data.drop(["season", "tournament name", "DK_Total_FP"], axis=1)
    new = new.dropna()
    new = new.set_index(['player']) 
    data = data[data["tournament name"].isin(["The Players Championship"])]
    new = new.groupby(level=0).mean()

    data = data.set_index(['player'])
    data = data.drop(data.columns.difference(['DK_Total_FP']), 1)


    new = new.join(data)
    new = new.reset_index()
    new = new.dropna() 
    print(new)
    new.to_csv('past_players_season_stats.csv', index=False, header=False, mode='a')


def main():
    names = currentPlayerNames()
    cleanCurrentData(names)
    for year in ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]:
        getSeason(year)

main()
