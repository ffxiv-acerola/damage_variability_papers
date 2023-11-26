"""
Helper module to perform retrospective log analysis (rla) from FFLogs.

Sends API requests to FFLogs to get damage actions for a job from a fight.
Transforms the response into human-readable data by 
  (i) Maps potencies to actions and buff types/amounts to buffs
  (ii) Convert into and action dataframe and rotation dataframe,
       the latter of which can be read into `ffxiv_stats`.

"""

import requests
import json
import time
import os

import numpy as np
import pandas as pd

from ffxiv_stats import Rate

# API config
url = "https://www.fflogs.com/api/v2/client"
api_key = os.environ['FFLOGS_TOKEN'] # or copy/paste your key here

headers = {
    'Content-Type': "application/json",
    'Authorization': f"Bearer {api_key}"
}

### Data ###
# Ability IDs
with open('abilities.json', 'r') as f:
    abilities = json.load(f)

action_potencies = pd.read_csv('dps_potencies.csv')
int_ids = np.asarray(list(abilities.keys()), dtype=int).tolist()
abilities = dict(zip(int_ids, abilities.values()))

crit_buffs = {
    'Chain Stratagem': 0.1,
    'Battle Litany': 0.1,
    "Wanderer's Minuet": 0.02,
    "Devilment": 0.2
    }

dh_buffs = {
    "Battle Voice": 0.2,
    "Devilment": 0.2
    }

abbreviations = {
    "Mug": "Mug",
    "Vulnerability Up": "Mug",
    "Arcane Circle": "AC",
    "Battle Litany": "BL",
    "Brotherhood": "BH",
    "Left Eye" : "LH",

    "Battle Voice": "BV",
    "Radiant Finale 1x": "RF1",
    "Wanderer's Minuet": "WM",
    "Technical Finish": "TF",
    "Devilment": "DV",
    
    "Embolden": "EM",
    "Searing Light": "SL",

    "Chain Stratagem": "CS",
    "Divination": "DIV",

    'Medicated': "Pot",

    "No Mercy": "NM"
}

def get_pull_dps_actions_from_fflogs(fight_code, fight_id, job, start_time=0, end_time=int(time.time())):
    actions = []
    variables = {"code": fight_code, "id": [fight_id], "job": job, "startTime": start_time, "endTime": end_time}
    next_timestamp = 0
    
    while next_timestamp is not None:    
        json_payload = {"query": """
        query DpsActions($code: String!, $id: [Int]!, $job: String!, $endTime: Float!, $startTime: Float!){
                reportData{
                    report(code: $code){
                    events(fightIDs:$id
                            startTime:$startTime, 
                            endTime:$endTime,
                            dataType:DamageDone,
                            sourceClass:$job,
                            viewOptions:1){
                        data
                        nextPageTimestamp
                    }    
                }
            }
        }
        """,
        "variables": variables,
        "operationName":"DpsActions"
        }
        
        r = requests.post(url=url, json=json_payload, headers=headers)
        data = r.text
        next_timestamp = json.loads(data)['data']['reportData']['report']['events']['nextPageTimestamp']
        actions.extend(json.loads(data)['data']['reportData']['report']['events']['data'])
        variables = {"code": fight_code, "id": [fight_id], "job": job, "startTime": next_timestamp, "endTime": end_time}

    return actions

def create_action_df(actions, crit_stat, dh_stat, medication_amt=262, medicated_buff_offset=0.05):
    """
    Turn the actions response from FFLogs API into a dataframe of actions.
    This serves as the precursor to the rotation dataframe, which is grouped by unique action and counted.
    """
    actions_df = pd.DataFrame(actions)
    # Only keep the "prepares action" or dot ticks
    actions_df = actions_df[
        (actions_df['type'] == 'calculateddamage') 
        | (actions_df['type'] == 'damage') 
        & (actions_df['tick'])
    ]

    # Buffs column won't show up if nothing is present, so make one with nans
    if 'buffs' not in pd.DataFrame(actions).columns:
        actions_df['buffs'] = np.NaN
        actions_df['buffs'] = actions_df['buffs'].astype('object')

    actions_df['ability_name'] = actions_df['abilityGameID'].replace(abilities)

    # Filter/rename columns
    actions_df = actions_df[
        ['timestamp', 'type', 'sourceID', 'targetID', 'abilityGameID', 
        'ability_name', 'buffs', 'amount', 'tick', 'multiplier']
    ].rename(columns={'buffs': 'buff_names'})

    # Add (tick) to a dot tick so the base ability name for 
    # application and ticks are distinct - e.g., Dia and Dia (tick)
    actions_df['ability_name'] = np.select(
        [
            actions_df['tick'] == True
        ],
        [
            actions_df['ability_name'] + ' (tick)'
        ],
        default=actions_df['ability_name']
    )

    # Split up buffs
    # I forgot why I replaced nan with -11, but it's probably important
    buffs = [str(x)[:-1].split('.') for x in actions_df['buff_names'].replace({np.NaN: -11})]
    buffs = [list(map(int, x)) for x in buffs]
    buffs = [[abilities[b] for b in k] for k in buffs]

    # Start to handle hit type buffs + medication
    r = Rate(crit_stat, dh_stat)

    multiplier = actions_df['multiplier'].tolist()
    name = actions_df['ability_name'].tolist()

    main_stat_adjust = [0] * len(buffs)
    crit_hit_rate_mod = [0] * len(buffs)
    direct_hit_rate_mod = [0] * len(buffs)
    p = [0] * len(buffs)

    # Loop over buffs/pots to create adjustments to rates/main stat
    # This is probably able to be done more efficiently with broadcasting/lambda functions
    # But it's fast enough and far from the most computationally expensive step
    for idx, b in enumerate(buffs):
        for s in b:
            if s in crit_buffs.keys():
                crit_hit_rate_mod[idx] += crit_buffs[s]

            elif s in dh_buffs.keys():
                direct_hit_rate_mod[idx] += dh_buffs[s]

            elif s == 'Medicated':
                main_stat_adjust[idx] += medication_amt
                multiplier[idx] -= medicated_buff_offset

            # TODO: need to handle auto CH/DH
            # TODO: handle pets

        # Create a unique action name based on the action + all buffs present 
        if None not in b:
            short_b = list(map(abbreviations.get, b))
            name[idx] = name[idx] + '-' + '_'.join(sorted(short_b))

        # FFlogs is nice enough to give us the overal damage multiplier
        multiplier[idx] = np.array([multiplier[idx]])
        p[idx] = r.get_p(crit_hit_rate_mod[idx], direct_hit_rate_mod[idx])

    # Assemble the action dataframe
    # Later we can groupby/count to create a rotation dataframe
    actions_df['buff_names'] = buffs
    actions_df['multiplier'] = multiplier
    actions_df['action_name'] = name
    actions_df['p'] = p
    actions_df['main_stat_add'] = main_stat_adjust
    actions_df['l_c'] = r.crit_dmg_multiplier()
    return actions_df

def create_rotation_df(actions_df, action_potencies=action_potencies):
    # Count how many times each action is performed
    value_counts = pd.DataFrame(
        actions_df[['action_name']].value_counts()
    ).reset_index().rename(columns={'count':'n'})

    # Get potencies and rename columns
    rotation_df = (
        actions_df.merge(value_counts, on='action_name')
                    .merge(action_potencies, left_on='abilityGameID', right_on='ability_id')
                    .drop_duplicates(subset=['action_name', 'n'])
                    .sort_values('n', ascending=False)
                    .rename(columns={'multiplier': 'buffs', "ability_name_x": "base_action"})
                    .reset_index(drop=True)
    )[['action_name', 'base_action', 'n', 'p', 'buffs', 'l_c', 'main_stat_add', 'potency', 'damage_type']]
    return rotation_df.sort_values('action_name')