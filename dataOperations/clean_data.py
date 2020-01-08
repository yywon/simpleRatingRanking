import pymongo
import json
import sys

responseCount = 22

url = 'mongodb://localhost:27017/'

client = pymongo.MongoClient(url)
db = client['ratingsrankingsframes']
usersCol = db['users']
responsesCol = db['responses']
batchesCol = db['batches']

completed_users = []

userRemove = 0

args = len(sys.argv) - 1

for user in usersCol.find():

    key2pay = user["key2pay"]
    userName = user["user"]
    indexes = user["indexes"]

    responseCount = responsesCol.count({'user' : userName})
     
    if(key2pay is None):
	key2pay = "none";

    print(userName + ": " + str(responseCount) + " responses. Key2pay: " + key2pay)
    
    if(responseCount >= 22):
        completed_users.append(userName)
    else:
        userRemove += 1
	    if(args > 0):
		    if(sys.argv[1] == "delete"):
        		responsesCol.remove({'user' : userName})
			    usersCol.remove({'user' : userName})
                #assign back to beginning
                for i in range(len(indexes)):
                    batch = batchesCol.find_one({'size': indexes[i][0], 'number': indexes[i][1]})
                    batch["assignedQuestions"][indexes[2]] = 0

print("Completed Users: " + str(len(completed_users)))
print("Users Removed: " + str(userRemove))



