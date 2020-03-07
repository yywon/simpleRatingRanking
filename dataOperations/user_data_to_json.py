import pymongo
import json
import time
import datetime
import sys

#url = 'mongodb://10.218.105.218:27017/'
url = 'mongodb://localhost:27017/'

dbase = sys.argv[1]

#set up database and all columns 
client = pymongo.MongoClient(url)
db = client[dbase]
usersCol = db['users']
responsesCol = db['responses']

dataArray = []

#iterate over users
for user in usersCol.find():
	dataArray.append(user)

file_name = "users" + dbase + ".json" 

with open(str(file_name), 'w+') as outfile:
	json.dump(dataArray, outfile) 


