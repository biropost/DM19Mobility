from os import listdir, path
import pandas as pd
from datetime import datetime
import re

whitelist_ids = {5, 46, 74, 127, 128, 129, 131, 165, 99, 105, 107, 108, 109, 152, 202, 203, 13, 28, 43, 52, 145, 146, 22, 24, 26, 42, 102, 103, 147, 214, 215, 19, 134, 137, 139, 142, 220, 222, 223, 224, 26, 248, 246, 245, 241, 84, 93, 235, 236, 33, 36, 37, 40, 49, 120, 113, 114, 115, 116, 117, 218, 219, 226, 227, 240, 21, 32, 38, 70, 95, 199, 78, 82, 83, 160, 161, 208, 209, 55, 9, 35, 88, 210, 211, 217, 72, 47, 201, 204, 91, 110, 228, 233, 234, 237}


def filter_whitelisted_trips(trip):
    # skip files that are not a trip
    if re.search("^\d+_\d+_\d{4}-\d{2}-\d{2}T\d{6}\.\d{1,3}$", trip) is None:
        return False
    # extract the trip id
    tid = int(trip.split('_')[1])
    # keep trips that are whitelisted
    if tid in whitelist_ids:
        return True
    # otherwise: skip the trip
    return False


# parse time to normalized ms
def replace_date(s):
    return s.group(0).replace('Z', '') + '.000Z'


def format_time(m):
    m = [re.sub(r':\d\d+Z', replace_date, sample) for sample in m]
    m = [float(datetime.strptime(sample, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%s.%f')) for sample in m]
    return m


data_dir = "data"
all_trips = listdir(data_dir)

whitelisted_trips = list(filter(filter_whitelisted_trips, all_trips))

df = None

processed = 0
total = len(whitelisted_trips)

for trip in whitelisted_trips:
    user_id = re.search('\d+', trip).group(0)
    trip_id = re.search('_\d+_', trip).group(0)
    trip_id = re.search('\d+', trip_id).group(0)
    processed = processed + 1
    print("processing user: ", user_id, ", trip: ", trip_id, " (", processed, "/", total, ")")

    # load markers data
    col_names = ["value", "key", "time", "mode", "col5", "col6", "col7", "station", "col9"]
    markers = pd.read_csv(path.join(data_dir, trip, 'markers.csv'), sep=';', names=col_names, skiprows=4)
    markers = markers.drop(['value', 'key', 'col5', 'col6', 'col7', 'col9'], axis=1)
    markers.drop(markers.tail(1).index, inplace=True)
    markers['time'] = format_time(markers['time'])
    markers['time'] = markers['time'].astype(int)
    markers = markers.set_index(['time'])
    markers.index = pd.to_datetime(markers.index, unit='ms')

    # load acceleration data
    acceleration = pd.read_csv(path.join(data_dir, trip, 'acceleration.csv'))
    acceleration['time'] = format_time(acceleration['time'])
    acceleration['time'] = acceleration['time'].astype(int)
    # down-sample acceleration data
    acceleration = acceleration.set_index(['time'])
    acceleration.index = pd.to_datetime(acceleration.index, unit='ms')
    acceleration = acceleration.resample('1L').mean()

    # combine acceleration and markers data
    df_trip = acceleration.merge(markers, on="time", how='left')
    df_trip = df_trip.ffill()

    # eliminate bimodal segments
    df_trip = df_trip.reset_index()
    drop_multimodal = []
    for i in range(int(df_trip.shape[0] / 10)):
        counts = df_trip.iloc[i * 10:i * 10 + 10].groupby('mode').count()
        if counts.shape[0] != 1:
            for val in range(i * 10, i * 10 + 10):
                drop_multimodal.append(val)
    df_trip.drop(drop_multimodal, inplace=True)

    # add ids and add to overall df
    df_trip['user_id'] = user_id
    df_trip['trip_id'] = trip_id
    if df is None:
        df = df_trip.copy()
    else:
        df = df.append(df_trip, ignore_index=True, sort=False)

df = df.set_index(['time'])
df.to_csv('export.csv', header=True)
