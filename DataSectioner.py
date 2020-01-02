from os import listdir, path
import pandas as pd
from datetime import datetime
import numpy as np
import re
import sklearn
import matplotlib.pyplot as plt

data_dir = "data"
all_trips = listdir(data_dir)
whitelist_ids = [5, 46, 74, 127, 128, 129, 131, 165, 99, 105, 107, 108, 109, 152, 202, 203, 13, 28, 43, 52, 145, 146, 22, 24, 26, 42, 102, 103, 147, 214, 215, 19, 134, 137, 139, 142, 220, 222, 223, 224, 26, 248, 246, 245, 241, 84, 93, 235, 236, 33, 36, 37, 40, 49, 120, 113, 114, 115, 116, 117, 218, 219, 226, 227, 240, 21, 32, 38, 70, 95, 199, 78, 82, 83, 160, 161, 208, 209, 55, 9, 35, 88, 210, 211, 217, 72, 47, 201, 204, 91]

def filter_whitelisted_trips(trip):
    # skip files that are not a trip
    if re.search("^\d+_\d+_\d{4}-\d{2}-\d{2}T\d{6}\.\d{3}$", trip) == None:
        return False
    # extract the trip id
    trip_id = trip.split('_')[1]
    # keep trips that are whitelisted
    if (int(trip_id) in whitelist_ids):
        return True
    # otherwise: skip the trip
    return False

whitelisted_trips = filter(filter_whitelisted_trips, all_trips)
my_list = list(whitelisted_trips)

# parse time to normalized ms
def dateReplace(s):
    replaced = s.group(0).replace('Z', '') + '.000Z'
    return replaced


def formatTime(m):
    m = [re.sub(r':\d\d+Z', dateReplace, sample) for sample in m]
    m = [float(datetime.strptime(sample, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%s.%f')) for sample in m]
    return m

df = None

for trip in my_list:
    user_id = re.search('\d+', trip).group(0)
    trip_id = re.search('_\d+_', trip).group(0)
    print(user_id, trip_id)
    path_markers = path.join(data_dir, trip, 'markers.csv')
    col_names = ["value", "key", "time", "mode", "col5", "col6", "col7", "station", "col9"]
    markers = pd.read_csv(path_markers, sep=';', names=col_names, skiprows=4)
    markers = markers.drop(['value', 'key', 'col5', 'col6', 'col7', 'col9'], axis=1)
    markers.drop(markers.tail(1).index, inplace=True)

    path_acc = path.join(data_dir, trip, 'acceleration.csv')
    acceleration = pd.read_csv(path_acc, sep=',')

    path_activity = path.join(data_dir, trip, 'activity_records.csv')
    activity = pd.read_csv(path_activity, sep=',')

    acceleration['time'] = formatTime(acceleration['time'])
    acceleration['time'] = acceleration['time'].astype(int)
    markers['time'] = formatTime(markers['time'])
    markers['time'] = markers['time'].astype(int)


    # combine acc and markers
    df_user = acceleration.merge(markers, on='time', how='inner')
    df_user = df_user.ffill()

    # drop n remaining rows for the 10 value segments
    n, m = df_user.shape
    rest = n % 10
    df_user.drop(df_user.tail(rest).index, inplace=True)
    df_user['user_id'] = user_id
    df_user['trip_id'] = trip_id

    if df is None:
        df = df_user.copy()
    else:
        df = df.append(df_user, ignore_index=True, sort=False)

print(df.head())
export_csv = df.to_csv('export_mobility.csv', index=None, header=True)