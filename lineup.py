from draftfast import rules
from draftfast.optimize import run, run_multi
from draftfast.orm import Player
from draftfast.csv_parse import salary_download
from draftfast.csv_parse import uploaders
from os import environ

GOLFER = 'G'
downloads = environ.get('downloads')


def main():
    cleanForLineups()
    player_pool = []
    exposures = []
    players = open("projections.csv", "r")
    for player in players:
        player = player.split(",")
        player_pool.append(Player(name=str(player[0]), cost=int(player[1]), proj=float(player[2]), pos=GOLFER))
        player_exposure = {}
        player_exposure["name"] = player[0]
        player_exposure["max"] = .6
        player_exposure["max"] = .6
        exposures.append(player_exposure)


    roster = run_multi(
        iterations=20,
        exposure_bounds=[
        {
            'name': 'Russell Henley',
            'min': 0,
            'max': 0.1,
        },
        {
            'name': 'Cameron Tringale',
            'min': 0,
            'max': 0.3,
        },
        {
            'name': 'Ryan Palmer',
            'min': 0,
            'max': 0.1,
        },
        {
            'name': 'Tyrrell Hatton',
            'min': 0,
            'max': 0.7,
        },
        {
            'name': 'Zach Johnson',
            'max': 0,
            'min': 0,
        },
        {
            'name': 'Rory McIlroy',
            'max': 0.7,
            'min': 0,
        },
        {
            'name': 'Patrick Cantlay',
            'max': 0.8,
            'min': 0,
        },
        {
            'name': 'Webb Simpson',
            'max': 0.8,
            'min': 0,
        },
        {
            'name': 'Joaquin Niemann',
            'max': 0.9,
            'min': 0,
        },
        {
            'name': 'Viktor Hovland',
            'max': 0.8,
            'min': 0,
        },
        {
            'name': 'Christiaan Bezuidenhout',
            'max': 0.55,
            'min': 0,
        },
        {
            'name': 'Brian Harman',
            'max': 0.2,
            'min': 0,
        },
        {
            'name': 'Charley Hoffman',
            'max': 0.1,
            'min': 0,
        },
        {
            'name': 'Patrick Reed',
            'max': 0.7,
            'min': 0,
        },
        {
            'name': 'Hideki Matsuyama',
            'max': 0.7,
            'min': 0,
        },
        {
            'name': 'Bernd Wiesberger',
            'max': 0.3,
            'min': 0,
        },
        {
            'name': 'Tony Finau',
            'max': 0.8,
            'min': 0,
        },
        {
            'name': 'Doug Ghim',
            'max': 0,
            'min': 0,
        },
        {
            'name': 'Shane Lowry',
            'max': 0,
            'min': 0,
        },

    ],
        rule_set=rules.DK_PGA_RULE_SET,
        player_pool=player_pool,
        verbose=True,
    )


    uploader = uploaders.DraftKingsNBAUploader(
        pid_file='./DKUpload.csv',
    )
    uploader.write_rosters(roster)



def cleanForLineups():
    player_to_salary = {}
    salaries = open("DKSalaries.csv", "r")
    for line in salaries:
        line = line.split(",")
        player_to_salary[line[2]] = line[5]
    salaries.close()

    projections = open("final_result.csv", "r")
    output = open("projections.csv", "w")

    for line in projections:
        line = line.split(",")
        if line[0] in player_to_salary:
            output.write(line[0] + "," + player_to_salary[line[0]] + "," + line[1])

main()