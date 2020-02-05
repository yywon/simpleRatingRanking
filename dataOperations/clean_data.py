import pymongo
import json
import sys

responseCount = 20

#url = 'mongodb://localhost:27017/'
url = 'mongodb://10.218.105.218:27017/'

dbase = sys.argv[1]

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
    
    if(responseCount >= 20):
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



