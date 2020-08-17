import pymongo
import json
import time
import datetime
import sys

#url = 'mongodb://10.218.105.218:27017/'
url = 'mongodb://localhost:27017/'

dbase = sys.argv[1]

#set up database and all columns 
client = pymongo.MongoClient(url)
db = client[dbase]
usersCol = db['users']
responsesCol = db['responses']

dataArray = []

#iterate over users
for user in usersCol.find():

	#get username
	userName = user['user']
	print(userName)
	#get questions
	questions = user['questions']

	#get the users responses for each question 
	for i in range(1,5):

		ranking = []
		rating = []

		#get ground truth for specified question
		questionOrder = questions[i-1]

		rankResponse = responsesCol.find_one({"user": userName, "collection": str(i), "type": "ranking"})
		batch = int(rankResponse["batch"])
		frames = int(rankResponse["frames"])
        
		rank = rankResponse["ranking"]
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


		#change rank to format
		pos = 1
		for ra in ranking:
			max_index = ranking.index(max(ranking))
			ranking[max_index] = pos
			pos += 1 

		#get ratings
		ratingResponse = responsesCol.find_one({"user": userName, "collection": str(i), "type": "rating"})
		ratingResponse = ratingResponse["estimates"]
		for i in ratingResponse:
			rating.append(int(float(filter(lambda x: x.isdigit(), i))))

		decoded_ratings = [None] * len(rating)
		#decode rating response
		for i in range(len(rating)):
			rankpos = ranks_min[i]
			cur_rat = rating[i]
			decoded_ratings[rankpos] = cur_rat 


		question = {
			"batch": batch,
			"frames": frames,
 			"rankings": ranking,
			"ratings": decoded_ratings,
			"groundtruth": questionOrder
		}

		dataArray.append(question)


rightNow = datetime.datetime.today().strftime('%m-%d-%Y')
file_name = rightNow + dbase + ".json" 

with open(str(file_name), 'w+') as outfile:
	json.dump(dataArray, outfile) 
