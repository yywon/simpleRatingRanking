import json
import numpy
from scipy.stats.stats import pearsonr  

with open('time_analysis.json') as f:
    data = json.load(f)

dots_dif = []
time = []

for response in data:
    predicted = int(response["rating"])
    actual = int(response["groundtruth"])
    rtime = int(response["time"])
    diff = abs(actual - predicted)
    dots_dif.append(diff)
    time.append(rtime)

print(numpy.corrcoef(dots_dif,time))
print(pearsonr(dots_dif,time))

