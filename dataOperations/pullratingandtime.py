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

		#get ground truth for specified question
		questionOrder = questions[i-1]

		#get pictures
		for j in range(len(questionOrder)):
			
		        gtruth = questionOrder[j]
			ratingResponse = responsesCol.find_one({"user": userName, "picture": str(j), "collection": str(i), "type": "rating"})
			rating = ratingResponse["estimate"]
			rating = int(float(rating))
            		timespent = int(ratingResponse["time"])

		    	question = {
		            "user": userName,
			    "time": timespent,
			    "rating": rating,
			    "groundtruth": gtruth
		    	}

		    	dataArray.append(question)

rightNow = datetime.datetime.today().strftime('%m-%d-%Y')
file_name = rightNow + dbase + ".json" 

with open(str(file_name), 'w+') as outfile:
	json.dump(dataArray, outfile) 
