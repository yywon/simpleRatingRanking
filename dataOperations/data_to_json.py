import json
from scipy.spatial import distance

gtruth = [50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81]


def findPosition(num, groundtruth):
    found = None
    for i in range(len(groundtruth)):
        for j in range(len(groundtruth[i])):
            if int(num) == int(groundtruth[i][j]):
                return i,j

with open("responseData 12 - 4.json", "r") as read_file:
    data = json.load(read_file)

rank_dif = 0
rate_dif = 0
rank_ui = 0
rate_ui = 0
count = 0
responseCount = 0
avg_ratings = [0] * 32

for response in data:
    groundtruth = response["groundtruth"]
    rating = response["ratings"]

    #get counts for usability scores
    if None not in(response["rank_dif"],response["rate_dif"],response["rank_ui"],response["rate_ui"]):
        rank_dif += int(response["rank_dif"])
        rate_dif += int(response["rate_dif"])
        rank_ui += int(response["rank_ui"])
        rate_ui += int(response["rate_ui"])

        count += 1
    
    for i in range(len(gtruth)):
        x,y = findPosition(gtruth[i], groundtruth)
        avg_ratings[i] += rating[x][y]

    responseCount += 1



avg_ratings = [x / responseCount for x in avg_ratings]
print(avg_ratings)

rank_dif = rank_dif/count
rate_dif = rate_dif/count
rank_ui = rank_ui/count
rate_ui = rate_ui/count

print("count " + str(count))
print("rank_dif " + str(rank_dif))
print("rate_dif " + str(rate_dif))
print("rank_ui " + str(rank_ui)) 
print("rate_ui " + str(rate_ui))

vec = [45,46,46,47,47,48,48,49,49,50,50,51,51,52,52,53,53,54,54,55,55,56,56,57,57,58,58,59,59,60,60,61]

print(vec)
print(avg_ratings)

dst1 = distance.euclidean(vec, gtruth)
dst2 = distance.euclidean(avg_ratings, gtruth)

print(dst1)
print(dst2)