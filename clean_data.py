mport pymongo
import json

url = 'mongodb://localhost:27017/'

noiseLevels = [128,64,32,16,8,4,2,1]

client = pymongo.MongoClient(url)
db = client['ratingsrankingsbasic']
usersCol = db['users']
questionPoolCol = db['questionPool']
responsesCol = db['responses']

completed_users = []

userRemove = 0

for user in usersCol.find():

    key2pay = user["key2pay"]
    userName = user["user"]
    responseCount = responsesCol.count({'user' : userName})

    print(userName + ": " + str(responseCount))
    if(key2pay is not None and responseCount == 40):
        completed_users.append(userName)
    else:
        userRemove += 1
        responsesCol.remove({'user' : userName})
	usersCol.remove({'user' : userName})

print("Completed Users: " + str(len(completed_users)))
print("Users Removed: " + str(userRemove))








