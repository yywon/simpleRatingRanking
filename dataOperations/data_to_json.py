import pymongo
import json
import time
import datetime

def to_rank(n):
	return n - 49

url = 'mongodb://10.218.105.218:27017/'
#url = 'mongodb://localhost:27017/'

#set up database and all columns 
client = pymongo.MongoClient(url)
db = client['ratingsrankingsdistributed']
usersCol = db['users']
responsesCol = db['responses']

dataArray = []

#iterate over users
for user in usersCol.find():
	
	print(user)

	rankings = []
	ratings = []
	
	#get username
	userName = user['user']

	#get the users responses for each question 
	for i in range(1,5):

		#get ground truth for specified question
		questionOrder = ground_truth[i-1]

		pictures = []

		rankResponse = responsesCol.find_one({"user": userName, "collection": str(i), "type": "ranking"})
        
		if (rankResponse is None):
			rank = [0,0,0,0]
		else:
			rank = rankResponse["ranking"]
			rank = [int(x) for x in rank]
			newrank = map(to_rank,rank)
			print('rank ' + str(newrank))
		
		rankings.append(newrank)

		#get pictures
		for j in range(4):
			
			ratingResponse = responsesCol.find_one({"user": userName, "picture": str(j), "collection": str(i), "type": "rating"})

			if ratingResponse is None:
				ratingResponse = 0
			else:
				ratingResponse = ratingResponse["estimate"]
				ratingResponse = int(float(ratingResponse))
				pictures.append(ratingResponse)

		#sort ratings in order of ground truth
		for k in range(1, len(questionOrder)):
			key = questionOrder[k] 
			key2 = pictures[k]
			l = k-1

			while l>=0 and key < questionOrder[l]:
				questionOrder[l+1] = questionOrder[l]
				pictures[l+1] = pictures[l]
				l = l - 1
			pictures[l+1] = key2
			questionOrder[l+1] = key

		ratings.append(pictures)

	#block to fix my errors
	
	rank_dif = None
	rate_dif = None
	rank_ui = None
	rate_ui = None	

	if user["surveyResults"] is not None:
		item = user["surveyResults"]
		print(item)
		rank_dif, rate_dif, rank_ui, rate_ui = fixResults(item, rank_dif, rate_dif, rank_ui, rate_ui)
		
	
	data = {
		"rank_dif": rank_dif,
		"rate_dif": rate_dif,
		"rank_ui": rank_ui,
		"rate_ui": rate_ui,
 		"rankings": rankings,
		"ratings": ratings,
		"groundtruth": ground_truth
	}

	dataArray.append(data)


rightNow = datetime.datetime.today().strftime('%m-%d-%Y')
file_name = rightNow + ".json" 

with open(str(file_name), 'w+') as outfile:
	json.dump(dataArray, outfile) 