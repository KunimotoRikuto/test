#author;R.Kunimoto, TAKENAKA co.
#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import csv

path = "C:\\Users\\1500570\\Documents\\R\\WS\\dataset"
fn = "data1_bar.csv"
file = csv.reader(open(path+"\\"+fn,"r"))

per = 0
data = []
lefter = [-1,0,1]


for line in file:
    data.append(line)
print(data)

while per < 4:
    ep = 0
    while ep < 9:
        cha = 0
        print(str(ep)+"th starting")
        file = csv.reader(open(path+"\\"+fn,"r"))
        x = np.array([])
        ff = True
        while cha < 3:
            if ff == True:
                label_y = str(data[0][ep])
                ff = False
                continue
            else:
                x = np.append(x,float(data[3*per+cha+1][ep]))
                cha += 1
        print(x)
        """
        mu, sigma = 100, 15
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)\
        ax.hist(x, bins=50)
        ax.set_title('first histogram $\mu=100,\ \sigma=15$')
        ax.set_xlabel('x')
        ax.set_ylabel(label_y)
        fig.save(path+"\\sample"+str(ep)+".jpg")
        """
        # vari = np.var(x)
        # avr = np.average(x)
        # y = 1 / np.sqrt(2 * np.pi * vari ) * np.exp(-(x - avr) ** 2 / (2 * vari))
        plt.subplot(3,3,ep+1)
        plt.bar(left=lefter, height=x)
        # plt.plot(x,y)
        plt.title(label_y,fontsize=7)
        plt.tick_params(labelsize=7)
        ff = True
        ep += 1
    plt.savefig(path+"\\sample"+str(per)+".png")
    plt.close()
    per += 1

print("おわた")