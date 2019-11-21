import json

noiseLevels = [128,64,32,16,8,4,2,1]

with open("../datafiles/responseData 11-8.json", "r") as read_file:
    data = json.load(read_file)

    for level in noiseLevels:
        print("Noise Level: " + str(level))
        print('\n')
        
        rankings = []
        ratings = []
        rankingAverages = [0,0,0,0]
        ratingAverages = [0,0,0,0]
        responseCount = 0

        for response in data:
            if response['noiseLevel'] == level:
                ranking = response['ranking']
                rating = response['rating']

                rankings.append(ranking)
                ratings.append(rating)
                groundtruth = response['groundtruth']
                responseCount +=1
            
        for rank in rankings:
            for i in range(4):
                rankingAverages[i] += rank[i]
        for i in range(len(rankingAverages)):
            sum = rankingAverages[i]
            avg = sum / responseCount
            print("Average Ranking: "+ str(avg))

        for rate in ratings:
            for i in range(4):
                ratingAverages[i] += rate[i]
        for i in range(len(rankingAverages)):
            sum = ratingAverages[i]
            avg = sum / responseCount
            print("Ground Truth: " + str(groundtruth[i]) + " Average Rating: " + str(avg))

        print('\n')

            





    







