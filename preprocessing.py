import pandas as pd
import math
import warnings

warnings.filterwarnings('ignore')

print("Beginning Preprocessing Script")

not_using_years = [2002, 2003, 2004, 2005, 2006, 2007, 2020, 2025]

metrics_data = pd.read_csv('DATA_METRICS.csv', index_col = False)
drop_cols = ['Vulnerable Top 2 Seed?', 'Tournament Winner?', 'Tournament Championship?', 'Final Four?', 'Top 12 in AP Top 25 During Week 6?', 'Post-Season Tournament', 'Post-Season Tournament Sorting Index', 'DFP', 'NSTRate', 'RankNSTRate', 'OppNSTRate', 'RankOppNSTRate']
drop_cols2 = ['toremove', 'Avg Possession Length (Offense)', 'Avg Possession Length (Offense) Rank', 'Avg Possession Length (Defense)', 'Avg Possession Length (Defense) Rank', 'Since', 'Full Team Name']
drop_cols3 = ['Region', 'Current Coach', 'Mapped Conference Name', 'Short Conference Name']
metrics_data = metrics_data.drop(columns=drop_cols)
metrics_data = metrics_data.drop(columns=drop_cols2)
metrics_data = metrics_data.drop(columns=drop_cols3)
metrics_data = metrics_data[~metrics_data['Season'].isin(not_using_years)] # drop any season without results

results_data = pd.read_csv('DATA_RESULTS.csv', index_col = False)
results_data = results_data[~results_data['YEAR'].isin(not_using_years)] # drop 2025 season
results_data = results_data.drop(columns=['BY YEAR NO', 'BY ROUND NO', 'TEAM NO'])

tournament_keys = set() # key is {year}{team}
for index, row in results_data.iterrows():
    # print(f"{row['YEAR']}{row['TEAM']}")
    tournament_keys.add(f"{row['YEAR']}{row['TEAM']}")

drop_rows = []
for index, row in metrics_data.iterrows():
    metric_key = f"{row['Season']}{row['Team']}"
    if metric_key not in tournament_keys:
        drop_rows.append(index)
    if metric_key == '2021Hartford':
        drop_rows.append(index)
metrics_data = metrics_data.drop(drop_rows)
metrics_data.insert(loc = 0, column='Tournament Wins', value = None)

wins_per_key = {}
for key in tournament_keys:
    wins_per_key[key] = 0
for index, row in results_data.iterrows():
    current_key = f"{row['YEAR']}{row['TEAM']}"
    round = int(row['ROUND'])
    wins = 6 - int(math.log2(round))
    wins_per_key[current_key] = wins
#print(wins_per_key)

for index, row in metrics_data.iterrows():
    metric_key = f"{row['Season']}{row['Team']}"
    metrics_data.loc[index, 'Tournament Wins'] = wins_per_key[metric_key]

for index, row in metrics_data.iterrows():
    years_string = row['Active Coaching Length']
    metrics_data.loc[index, 'Active Coaching Length']  = int(years_string.split(' ')[0])

# tournament_teams = set()
# for index, row in results_data.iterrows():
#     tournament_teams.add = row['TEAM']
#     metrics_data.loc[index, 'TEAM'] = f"{row['TEAM']} {row['YEAR']}"
# tk_list = list(tournament_teams)
# metrics_data.insert(loc = 3, column='KEY', value = None)
# for index, row in results_data.iterrows():
#     team_val = tk_list.index(row['TEAM'])
#     metrics_data.loc[index, 'KEY'] = 0
# metrics_data.rename(columns={'YEAR': 'KEY'}, inplace=True)


# count = 0
# for index, row in metrics_data.iterrows():
#     for dex, val in enumerate(row):
#         strval = str(val)
#         # print(strval)
#         isnum = True
#         for c in strval:
#             if((not c.isnumeric()) and c != '.'):
#                 isnum = False
#         newval = ""
#         if(not isnum):
#             for c in strval:
#                 if(c.isalpha()):
#                     newval += c.lower()
#                 elif(c.isnumeric()):
#                     newval += c
#             # print(f"{count}, {dex}")
#             metrics_data.iloc[count, dex] = newval
#     count += 1

metrics_data.to_csv('PREPROCESSED_DATA.csv', index = False)
print("Finished!")