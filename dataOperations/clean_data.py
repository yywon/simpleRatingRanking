#script to only keep responses that are valid and tied to a user, as well as clear out users that have not finished

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
		    if indexes[i][1] > 3:
			    discard = 1
        if indexes[i][0] == 3:
		    if indexes[i][1] > 5:
			    discard = 1
        if indexes[i][0] == 5:
		    if indexes[i][1] > 9:
			    discard = 1
        if indexes[i][0] == 6:
		    if indexes[i][1] > 11:
			discard = 1

    #validate user responses
    for i in range(len(indexes)):

        question = str(indexes[i][2])
        update = {"$set": {}}
        update['$set']["assignmentStatus."+question] = 2

        batchesCol.update_many({'size': indexes[i][0], 'number': indexes[i][1]}, update)


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


#update all batches that have actually been assigned to 1, and those that havent to 0
real = 0
changed = 0
 
for batch in batchesCol.find():

    size = batch['size']
    number = batch['number']
    status = batch['assignmentStatus']

    for i in range(len(status)):

        assignment = str(status[i])

        if assignment is "2":
            real += 1

            update = {"$set": {}}
            update['$set']["assignmentStatus."+str(i)] = 1

            batchesCol.update_many({'size': size, 'number': number}, update)

        else:

            changed += 1

            update = {"$set": {}}
            update['$set']["assignmentStatus."+str(i)] = 0

            batchesCol.update_many({'size': size, 'number': number}, update)

print("Unfinished Users: ")
print("Completed Users: " + str(len(completed_users)))
print("Users Removed: " + str(userRemove))

print("Invalid Responses: ")
print("Number of assignments kept: ", str(real))
print("Number of assignments removed: ", str(changed))


