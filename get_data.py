import pymongo

url = "mongodb://rwkemmer@10.218.105.218:27017/"

noiseLevels = [128,64,32,16,8,4,2,1]

#set up database and all columns 
client = pymongo.MongoClient(url)
db = client.db('ratingsrankingsbasic')
usersCol = db.collection('users')
questionPoolCol = db.collection('questionPool')
responsesCol = db.collection('responses')

for user in usersCol.find():

    userQuestions = []
    userRankResponses = []
    userRatingResponses = []

    #retieve permutation
    userPermutation = user.group4answers
    print(userPermutation)

    #get username 
    userName = user.user

    #query from question pool and get the users assigned question
    for i in range(len(userPermutation)):

        #get noise level and permutation number
        variation = userPermutation[i]
        noiseLevel = noiseLevels[i]

        questionOrder = questionPoolCol.findOne({"noiselevel": noiseLevel, "variation": variation})
        print(questionOrder)

        #make array containing all user questions for each noise level
        userQuestions.append(questionOrder)

    #get the users responses for each question 
    for i in range(len(noiseLevels)):

        pictures = []

        #collection is i+1
        collection = i + 1 

        rankResponse = responsesCol.find({"user": userName, "collection": collection, "type": "ranking"})

        #get pictures
        for j in range(4):
            ratingresponse








