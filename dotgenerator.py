import numpy as np
import matplotlib.pyplot as plt
import os 
from itertools import permutations
import json

# Baseline data
Base = 50
area = np.pi*3

masterArray = []

noiseLevels = {1,2,4,8,16,32,64,128}

#loop through noise levels and generate pictures
for i in noiseLevels:

    #PATH = 'C:/Users/rwkemmer/Desktop/dots/'

    print("Noise Level: " + str(i))

    pictureData = []

    #os.chdir(PATH)
    #os.mkdir(str(i))

    #each picture per noise level
    for j in range(4):
        N = Base + (i * j)

        pictureData.append(N)

        existing_locations = []
        existing_locations.append(np.array([0,0]))
        dots = 0

        #add each individual dot
        while (dots < N):

            x = np.random.rand()
            y = np.random.rand()

            new_location = np.array([x,y])
            valid_checks = 0

            #check to ensure new dot is not too close to others
            for existing_location in existing_locations:
                vecdif = existing_location - new_location
                
                if ((abs(vecdif[0]) < .03) & (abs(vecdif[1]) < .03)):
                    #print("broke criteria")
                    break
                else:
                    valid_checks += 1

            if valid_checks == len(existing_locations):
                dots += 1
                existing_locations.append(new_location)    
                #plt.plot(x, y, 'ko')

        print("number of dots: " + str(dots))

        '''
        axes = plt.gca()
        axes.set_xlim([-0.01,1.01])
        axes.set_ylim([-0.01,1.01])

        plt.axes().set_aspect('equal')
        plt.axis('off')

        plt.savefig(PATH + str(i) + '/' + str(N),bbox_inches='tight', pad_inches=0)

        plt.clf()

        '''

    activityArray = [i, pictureData]

    masterArray.append(activityArray)

permutationArray = []

for i in masterArray:
    permutationsofActivity = []
    activity = i[0]
    pictureData = i[1]
    #print(pictureData)
    permutationsofActivity = permutations(pictureData)
    permy = []

    counter = 0
    for permutation in permutationsofActivity:
        perm = [activity, counter, permutation]
        print(perm)
        permutationArray.append(perm)
        counter += 1

data = []

for i in permutationArray: 
    ordering = i[2]
    instance = {
        'noiselevel': i[0],
        'variation': i[1],
        'array': ordering
    } 

    data.append(instance)

with open('public/data/questions.json', 'w') as outfile:
    json.dump(data, outfile)


'''
for i in permutationArray: 
    ordering = i[2]
    data.append({
        'noiselevel': i[0],
        'variation': i[1],
        'array': ordering,
        #'pos0': ordering[0],
        #'pos1': ordering[1],
        #'pos2': ordering[2],
        #'pos3': ordering[3]
    })

#dump to outfile
with open('public/data/questions.json', 'w') as outfile:
    json.dump(data, outfile)

'''




