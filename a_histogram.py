#author;R.Kunimoto, TAKENAKA co.
#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import csv

path = "C:\\Users\\1500570\\Documents\\R\\WS\\dataset"
fn = "data1_his.csv"


ep = 0
ff = True
#list = []
while ep < 9:
    print(str(ep)+"th starting")
    """
    x = 100 + 15 * np.random.randn(10000)
    print(x)
    """
    file = csv.reader(open(path+"\\"+fn,"r"))
    x = np.array([])
    for line in file:
        if ff == True:
            label_y = str(line[ep])
            print(label_y)
            ff = False
            continue
        else:
            x = np.append(x,float(line[ep]))
    print(x)
    """
    mu, sigma = 100, 15

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.hist(x, bins=50)
    ax.set_title('first histogram $\mu=100,\ \sigma=15$')
    ax.set_xlabel('x')
    ax.set_ylabel(label_y)
    fig.save(path+"\\sample"+str(ep)+".jpg")
    """
    vari = np.var(x)
    avr = np.average(x)
    y = 1 / np.sqrt(2 * np.pi * vari ) * np.exp(-(x - avr) ** 2 / (2 * vari))
    plt.subplot(3,3,ep+1)
    plt.hist(x)
    # plt.plot(x,y)
    plt.title(label_y,fontsize=7)
    plt.xlabel("fluc", fontsize=7)
    plt.ylabel("num_seq", fontsize=7)
    plt.tick_params(labelsize=7)
    ff = True
    print(str(ep)+"th ending")
    ep += 1
plt.savefig(path+"\\sample"+str(ep)+".png")