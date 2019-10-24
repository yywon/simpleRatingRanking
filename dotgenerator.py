import numpy as np
import matplotlib.pyplot as plt
import os 
from itertools import permutations
from itertools import combinations
import json

# Baseline data
Base = 50
area = np.pi*3

masterArray = []

PATH = 'C:/Users/rwkemmer/Desktop/dots/'

pictureData = []

os.chdir(PATH)

#each picture per noise level
for j in range(40):
    N = Base + j

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
            plt.plot(x, y, 'ko')

        
    axes = plt.gca()
    axes.set_xlim([-0.01,1.01])
    axes.set_ylim([-0.01,1.01])

    plt.axes().set_aspect('equal')
    plt.axis('off')

    plt.savefig(PATH + '/' + str(N),bbox_inches='tight', pad_inches=0)

    plt.clf()
