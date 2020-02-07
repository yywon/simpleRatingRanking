import pymongo
import json
import sys

responseCount = 20

url = 'mongodb://localhost:27017/'
#url = 'mongodb://10.218.105.218:27017/'

dbase = sys.argv[1]
print(dbase)

client = pymongo.MongoClient(url)
db = client[dbase]
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
        key2pay = "none"

    print(userName + ": " + str(responseCount) + " responses. Key2pay: " + key2pay)
    
    discard = 0
    #check if indexes are in valid range
    for i in range(4):
        if indexes[i][0] == 2:
		print('frame: 2 batch: ' + str(indexes[i][1]))
		if indexes[i][1] > 3:
			discard = 1
	if indexes[i][0] == 3:
		print('frame: 3 batch: ' + str(indexes[i][1])) 
		if indexes[i][1] > 5:
			discard = 1
	if indexes[i][0] == 5:
		print('frame: 5 batch: ' + str(indexes[i][1])) 
		if indexes[i][1] > 9:
			discard = 1
	if indexes[i][0] == 6:
		print('frame: 6 batch: ' + str(indexes[i][1])) 
		if indexes[i][1] > 11:
			discard = 1

    if(responseCount >= 20 and discard == 0):
        completed_users.append(userName)
    else:
        userRemove += 1
        if(args > 1):
            if(sys.argv[2] == "delete"):

                #remove assignment from batch 
                for i in range(len(indexes)):

                    question = str(indexes[i][2])
                    update = {"$set": {}}
                    update['$set']["assignmentStatus."+question] = 0

                    batchesCol.update_one({'size': indexes[i][0], 'number': indexes[i][1]}, update)
                
                responsesCol.delete_many({'user' : userName})
                usersCol.delete_one({'user' : userName})

print("Completed Users: " + str(len(completed_users)))
print("Users Removed: " + str(userRemove))



