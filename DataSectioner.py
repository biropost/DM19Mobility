from os import listdir, path
import pandas as pd
from datetime import datetime
import numpy as np
import re
import sklearn
import matplotlib.pyplot as plt

whitelist_ids = [5, 46, 74, 127, 128, 129, 131, 165, 99, 105, 107, 108, 109, 152, 202, 203, 13, 28, 43, 52, 145, 146, 22, 24, 26, 42, 102, 103, 147, 214, 215, 19, 134, 137, 139, 142, 220, 222, 223, 224, 26, 248, 246, 245, 241, 84, 93, 235, 236, 33, 36, 37, 40, 49, 120, 113, 114, 115, 116, 117, 218, 219, 226, 227, 240, 21, 32, 38, 70, 95, 199, 78, 82, 83, 160, 161, 208, 209, 55, 9, 35, 88, 210, 211, 217, 72, 47, 201, 204, 91]

data_dir = "cgt-stud"

trips = listdir(data_dir)

trip = trips[0]

path_markers = path.join(data_dir, trip, 'markers.csv')
col_names = ["time", "key", "value", "col4", "col5", "col6", "col7", "col8", "col9"]
markers = pd.read_csv(path_markers, sep=';', names=col_names, skiprows=1)

path_acc = path.join(data_dir, trip, 'acceleration.csv')
acceleration = pd.read_csv(path_acc, sep=',')


# parse time to normalized ms
def dateReplace(s):
    replaced = s.group(0).replace('Z', '') + '.000Z'
    return replaced
def formatTime(m):
    m = np.array(m)
    m = [re.sub(r':\d\d+Z', dateReplace, sample) for sample in m]
    m = [float(datetime.strptime(sample, "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%s.%f')) for sample in m]
    return m

acceleration['time'] = formatTime(acceleration['time'])
markers['time'] = formatTime(markers['time'])

# combine acc and markers
set = acceleration.join(markers, on='time', how='left', lsuffix='_left', rsuffix='_right')
print(set.head())
