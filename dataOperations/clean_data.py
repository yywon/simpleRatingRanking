import pymongo
import json

url = 'mongodb://localhost:27017/'

client = pymongo.MongoClient(url)
db = client['ratingsrankingsdistributed']
usersCol = db['users']
responsesCol = db['responses']

completed_users = []

userRemove = 0

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
        responsesCol.remove({'user' : userName})
	usersCol.remove({'user' : userName})

print("Completed Users: " + str(len(completed_users)))
print("Users Removed: " + str(userRemove))








