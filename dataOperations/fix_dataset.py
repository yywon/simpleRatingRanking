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

#update all batches that have actually been assigned to 2
for user in usersCol.find():

    userName = user["user"]
    indexes = user["indexes"]

    for i in range(len(indexes)):

        question = str(indexes[i][2])
        update = {"$set": {}}
        update['$set']["assignmentStatus."+question] = 2

        batchesCol.update_one({'size': indexes[i][0], 'number': indexes[i][1]}, update)
    
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
            update['$set']["assignmentStatus."+assignment] = 1

            batchesCol.update_one({'size': size, 'number': number}, update)

        else:

            changed += 1

            update = {"$set": {}}
            update['$set']["assignmentStatus."+assignment] = 0

            batchesCol.update_one({'size': size, 'number': number}, update)


print("number of assignments kept:", str(real))
print("number of assignments removed:", str(changed))




