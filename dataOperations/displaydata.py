import json

gtruth = [128,64,32,16,8,4,2,1]

with open("responseData.json", "r") as read_file:
    data = json.load(read_file)

rank_dif = 0
rate_dif = 0
rank_ui = 0
rate_ui = 0
count = 0

for response in data:
    groundtruth = response["groundtruth"]

    print(response["rank_dif"])

    
    rank_dif += int(response["rank_dif"])
    rate_dif += int(response["rate_dif"])
    rank_ui += int(response["rank_ui"])
    rate_ui += int(response["rate_ui"])

    count += 1

rank_dif = rank_dif/count
rate_dif += rate_dif
rank_ui += rank_ui
rate_ui += rate_ui

print("rank_dif" + rank_dif)
print("rate_dif" + rate_dif)
print("rank_ui" + rank_ui)
print("rate_ui" + rate_ui)

    







