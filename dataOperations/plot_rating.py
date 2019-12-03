import matplotlib.pyplot as plt
import json

noiseLevels = [128,64,32,16,8,4,2,1]

with open("../datafiles/responseData 11-8.json", "r") as read_file:
    data = json.load(read_file)

ratings = []

count = 0
ratings = []
rate = []
i = 0
while i < len(data):
    print(data[i])
    rate.append(data[i]["rating"][0])
    count += 1
    i += 1
    if(count % 8 == 0):
        count = 0
        ratings.append(rate)
        rate = []
        print(rate)
        plt.plot(rate)

plt.show()