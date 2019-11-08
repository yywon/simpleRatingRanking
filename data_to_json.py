import pymongo
import json

url = 'mongodb://localhost:27017/';

noiseLevels = [128,64,32,16,8,4,2,1]

#set up database and all columns 
client = pymongo.MongoClient(url)
db = client['ratingsrankingsbasic']
usersCol = db['users']
questionPoolCol = db['questionPool']
responsesCol = db['responses']

dataArray = []
#iterate over users
for user in usersCol.find():

    #retrieve permutation
    userPermutation = user["group4Answers"]

    #get username
    userName = user['user']

    #get the users responses for each question 
    for i in range(len(noiseLevels)):
	
        pictures = []
	rank = []
        temp = [None] * 4
	
	#get noise level and permutation number
        variation = userPermutation[i]
        noiseLevel = noiseLevels[i]

        questionOrder = questionPoolCol.find_one({"noiselevel": noiseLevel, "variation": variation})
        questionOrder = questionOrder["array"]

        #collection is i+1
        collection = i + 1
	collection = str(collection)

        rankResponse = responsesCol.find_one({"user": userName, "collection": collection, "type": "ranking"})
	if (rankResponse is None):
		rank = [0,0,0,0]
	else:
		rank = rankResponse["ranking"]
		rank = [int(x) for x in rank]
		k = 4
        	while (k > 0):
            		maxpos = rank.index(max(rank))
            		temp[maxpos] = k
            		rank[maxpos] = -1
            		k -= 1
        rank = temp

        #get pictures
        for j in range(4):
	    picture = str(j)
            ratingResponse = responsesCol.find_one({"user": userName, "picture": picture, "collection": collection, "type": "rating"})
	    if ratingResponse is None:
		ratingResponse = 0
	    else:
	    	ratingResponse = ratingResponse["estimate"]
		ratingResponse = int(float(ratingResponse))
            pictures.append(ratingResponse)

	
	#sort ratings in order of ground truth
	for i in range(1, len(questionOrder)):
		key = questionOrder[i] 
		key2 = pictures[i]
		j = i-1

		while j>=0 and key < questionOrder[j]:
			questionOrder[j+1] = questionOrder[j]
			pictures[j+1] = pictures[j]
			j = j - 1
		pictures[j+1] = key2
		questionOrder[j+1] = key
	

       	data = {
			"noiseLevel" : noiseLevel,
 			"ranking": rank,
			"rating": pictures,
			"groundtruth": questionOrder
		}

	dataArray.append(data)
	
with open('responseData.json', 'w') as outfile:
	json.dump(dataArray, outfile) 

	

	

