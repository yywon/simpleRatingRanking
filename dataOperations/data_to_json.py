import pymongo
import json

url = 'mongodb://localhost:27017/'

#set up database and all columns 
client = pymongo.MongoClient(url)
db = client['ratingsrankingsdistributed']
usersCol = db['users']
questionPoolCol = db['questionPool']
responsesCol = db['responses']

dataArray = []

#iterate over users
for user in usersCol.find():

	rankings = []
	ratings = []

	ground_truth = user['group4Answers']
	#get username
	userName = user['user']

	#get the users responses for each question 
	for i in range(1,9):

		#get ground truth for specified question
		questionOrder = ground_truth[i-1]

		pictures = []

		rankResponse = responsesCol.find_one({"user": userName, "collection": str(i), "type": "ranking"})
		if (rankResponse is None):
			rank = [0,0,0,0]
		else:
			rank = rankResponse["ranking"]
			rank = [int(x) for x in rank]
			print('rank ' + str(rank))
		
		rankings.append(rank)

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
	if(user["surveyResults"]["ranking_difficulty"]):
		rank_dif = user["surveyResults"]["ranking_difficulty"]
	else:
		rank_dif = user["surveyResults"]["ranking_likeability"]

	if(user["surveyResults"]["rating_difficulty"]):
		rate_dif = user["surveyResults"]["rating_difficulty"]
	else:
		rate_dif = user["surveyResults"]["rating_like"]
	
	if(user["surveyResults"]["ranking_ui"]):
		rank_ui = user["surveyResults"]["ranking_ui"]
	else:
		rank_ui = user["surveyResults"]["rating_likeability"]

	if(user["surveyResults"]["rating_ui"]):
		rate_ui = user["surveyResults"]["rating_ui"]
	else:
		rate_ui = user["surveyResults"]["rating_expressiveness"]
		
	
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

with open('responseData.json', 'w') as outfile:
	json.dump(dataArray, outfile) 

	

	

