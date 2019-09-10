import numpy as np
import matplotlib.pyplot as plt
import os 

# Create data
Base = 200
colors = (0,0,0)
area = np.pi*3

noiseLevels = {6,10,14,18}

for i in noiseLevels:

    PATH = 'C:/Users/rwkemmer/Documents/apptemplate/dots/'

    print("Noise Level: " + str(i))

    os.chdir(PATH)
    os.mkdir(str(i))

    for j in range(4):
        N = 200 + (i * j)

        x = np.random.rand(N)
        y = np.random.rand(N)

        print(x.shape)
        print(y.shape)

        plt.scatter(x, y, s=area, c=colors, alpha=0.5)

        axes = plt.gca()
        axes.set_xlim([-0.01,1.01])
        axes.set_ylim([-0.01,1.01])

        plt.axes().set_aspect('equal')
        plt.axis('off')

        plt.savefig(PATH + str(i) + '/' + str(N),bbox_inches='tight', pad_inches=0)

        plt.clf()