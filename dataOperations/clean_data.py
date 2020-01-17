import pymongo
import json
import sys

responseCount = 22

#url = 'mongodb://localhost:27017/'
url = 'mongodb://10.218.105.218:27017/'

client = pymongo.MongoClient(url)
db = client['ratingsrankingsframes']
usersCol = db['users']
responsesCol = db['responses']
batchesColA = db['batchesA']
batchesColB = db['batchesB']

completed_users = []

userRemove = 0

args = len(sys.argv) - 1

for user in usersCol.find():

    key2pay = user["key2pay"]
    userName = user["user"]
    indexesA = user["indexesA"]
    indexesB = user["indexesB"]

    responseCount = responsesCol.count({'user' : userName})
     
    if(key2pay is None):
        key2pay = "none"

    print(userName + ": " + str(responseCount) + " responses. Key2pay: " + key2pay)
    
    if(responseCount >= 26):
        completed_users.append(userName)
    else:
        userRemove += 1
        if(args > 0):
            if(sys.argv[1] == "delete"):

                #remove assignment from batch A
                for i in range(len(indexesA)):

                    question = str(indexesA[i][2])
                    update = {"$set": {}}
                    update['$set']["assignmentStatus."+question] = 0

                    batchesColA.update_one({'size': indexesA[i][0], 'number': indexesA[i][1]}, update)

                #remove assignment from batch B
                for i in range(len(indexesB)):

                    question = str(indexesB[i][2])
                    update = {"$set": {}}
                    update['$set']["assignmentStatus."+question] = 0

                    batchesColB.update_one({'size': indexesB[i][0], 'number': indexesB[i][1]}, update)
                
                responsesCol.delete_many({'user' : userName})
                usersCol.delete_one({'user' : userName})

print("Completed Users: " + str(len(completed_users)))
print("Users Removed: " + str(userRemove))



