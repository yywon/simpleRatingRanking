import pymongo
import json
import sys

url = 'mongodb://localhost:27017/'

client = pymongo.MongoClient(url)
db = client['ratingsrankingsdistributed']
usersCol = db['users']
responsesCol = db['responses']

completed_users = []

userRemove = 0

args = len(sys.argv) - 1

for user in usersCol.find():

    key2pay = user["key2pay"]
    userName = user["user"]
    responseCount = responsesCol.count({'user' : userName})
     
    if(key2pay is None):
	key2pay = "none";

    print(userName + ": " + str(responseCount) + " responses. Key2pay: " + key2pay)
    
    if(responseCount >= 40):
        completed_users.append(userName)
    else:
        userRemove += 1
	if(args > 0):
		if(sys.argv[1] == "delete"):
        		responsesCol.remove({'user' : userName})
			usersCol.remove({'user' : userName})

print("Completed Users: " + str(len(completed_users)))
print("Users Removed: " + str(userRemove))



