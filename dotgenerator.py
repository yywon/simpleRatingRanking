import numpy as np
import matplotlib.pyplot as plt
import os 

# Baseline data
Base = 50
area = np.pi*3

noiseLevels = {1,2,4,8,16,32,64,128}

#loop through noise levels
for i in noiseLevels:

    PATH = 'C:/Users/rwkemmer/Desktop/dots/'

    print("Noise Level: " + str(i))

    os.chdir(PATH)
    os.mkdir(str(i))

    #each picture per noise level
    for j in range(4):
        N = Base + (i * j)

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

        print("number of dots: " + str(dots))

        axes = plt.gca()
        axes.set_xlim([-0.01,1.01])
        axes.set_ylim([-0.01,1.01])

        plt.axes().set_aspect('equal')
        plt.axis('off')

        plt.savefig(PATH + str(i) + '/' + str(N),bbox_inches='tight', pad_inches=0)

        plt.clf()