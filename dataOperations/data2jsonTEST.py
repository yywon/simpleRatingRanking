import pymongo
import json
import time
import datetime
import sys

#set up database and all columns 

with open('TESTDATA.json') as f:
    responsesCol = json.load(f)

with open('TESTDATAUSER.json') as f:
    usersCol = json.load(f)

dataArray = []

#iterate over users
for user in usersCol:

    #get username
    userName = user['user']
    print(userName)
    #get questions
    questions = user['questions']
    print(questions)

    #get the users responses for each question 
    for i in range(1,5):

        ranking = []
        rating = []

        #get ground truth for specified question
        questionOrder = questions[i-1]
        
        for resp in responsesCol:
            if resp["collection"] == str(i) and resp["type"] == "ranking":
                rankResponse = resp

        batch = int(rankResponse["batch"])
        frames = int(rankResponse["frames"])
        
        rank = rankResponse["ranking"]
        print(rank)
        ranking = [int(x) for x in rank]
        rank_copy = ranking[:]
        
        #get mins:
        pos = 0
        ranks_min = [None] * len(rank_copy)
        for ra in rank_copy:
            min_index = ranking.index(min(rank_copy))
            rank_copy[min_index] = 1000
            ranks_min[min_index] = pos
            pos += 1

        print(ranks_min)

        #change rank to format
        pos = 1
        for ra in ranking:
            max_index = ranking.index(max(ranking))
            ranking[max_index] = pos
            pos += 1 

        print(ranking)

        #get ratings
        for resp in responsesCol:
            if resp["collection"] == str(i) and resp["type"] == "rating":
                ratingResponse = resp

        ratingResponse = ratingResponse["estimates"]
        for i in ratingResponse:
            rating.append(int(i))

        print(rating)

        decoded_ratings = [None] * len(rating)

        #decode rating response
        for i in range(len(rating)):
            rankpos = ranks_min[i]
            cur_rat = rating[i]
            decoded_ratings[rankpos] = cur_rat 

        print(decoded_ratings)

        question = {
            "batch": batch,
            "frames": frames,
             "rankings": ranking,
            "ratings": decoded_ratings,
            "groundtruth": questionOrder
        }

        dataArray.append(question)


rightNow = datetime.datetime.today().strftime('%m-%d-%Y')

with open(str("TEST.json"), 'w+') as outfile:
    json.dump(dataArray, outfile) 
