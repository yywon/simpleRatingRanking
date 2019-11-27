import json

gtruth = [50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82]

with open("responseData.json", "r") as read_file:
    data = json.load(read_file)

rank_dif = 0
rate_dif = 0
rank_ui = 0
rate_ui = 0
count = 0

for response in data:
    groundtruth = response["groundtruth"]

    #get counts for usability scores
    if None not in(response["rank_dif"],response["rate_dif"],response["rank_ui"],response["rate_ui"]):
        rank_dif += int(response["rank_dif"])
        rate_dif += int(response["rate_dif"])
        rank_ui += int(response["rank_ui"])
        rate_ui += int(response["rate_ui"])

        count += 1
    
    for picture in gtruth:

rank_dif = rank_dif/count
rate_dif = rate_dif/count
rank_ui = rank_ui/count
rate_ui = rate_ui/count



print("count " + str(count))
print("rank_dif " + str(rank_dif))
print("rate_dif " + str(rate_dif))
print("rank_ui " + str(rank_ui))
print("rate_ui " + str(rate_ui))

    







