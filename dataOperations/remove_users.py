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

args = len(sys.argv)

badUsers =[]
for i in range(2, args):
	badUsers.append(sys.argv[i])


for user in usersCol.find():
    key2pay = user["key2pay"]
    userName = user["user"]
    indexes = user["indexes"]
    
    if userName in badUsers:

	print(userName)

        #remove assignment from batch 
        for i in range(len(indexes)):

            question = str(indexes[i][2])
            update = {"$set": {}}
            update['$set']["assignmentStatus."+question] = 0

            batchesCol.update_one({'size': indexes[i][0], 'number': indexes[i][1]}, update)
                
        responsesCol.delete_many({'user' : userName})
        usersCol.delete_one({'user' : userName})

	userRemove+=1

print("Users Removed: " + str(userRemove))



