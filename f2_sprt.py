#author;R.Kunimoto, TAKENAKA co.
#coding:utf-8

import numpy as np
import csv
import re
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
from sklearn import cluster
from collections import Counter
import random

def generate_random_color():
    return '{:X}{:X}{:X}'.format(*[random.randint(0, 255) for _ in range(3)])

colorbar = ["008b8b","008080","2f4f4f","006400","008000","228b22","2e8b57","3cb371","66cdaa","8fbc8f","7fffd4","98fb98"]

path = "C:\\Users\\1500570\\Documents\\R\\WS\\dataset_f2"
file = "170911_data_F2.csv"

mama = re.compile(u"前月")

pini1 = 0.2
pini0 = 1 -pini1
pouti1 = 0.8
pouti0 = 1 -pouti1
alp = 0.05
bet = 0.05
c1 =bet /(1-alp)
c2 =(1-bet)/alp


with open(path+"\\"+file, "r+") as f:
    reader = csv.DictReader(f)
    rf = reader.fieldnames
    rf2 = [] #not前月属性のインデクス配列
    rf3 = [] #NA許容範囲の属性名のインデクス配列
    p = 0
    for ff in rf:
        if mama.search(ff) == None:
            rf2.append(p)
        p += 1

    reader2 = csv.reader(f)
    mat1 = []
    for line in reader2:
        mat1.append(line)
    mat1 = np.array(mat1).T

    mat2 = np.empty((0,len(mat1[0]))) #rf2に準拠したデータ配列
    for i in rf2:
        mat2 = np.vstack((mat2, mat1[i]))
    print(rf2)

    mat3 = np.empty((0,len(mat1[0]))) #rf2かつNA許容なデータ配列
    mm = 0
    for line in mat2:
        num = len(np.where(line=="NA")[0])
        if num == 0:
            mat3 = np.vstack((mat3,line))
            rf3.append(rf2[mm])
        mm += 1

    cn = 1
    while cn <len(mat3):
        temp = mat3[cn][0:39].astype(np.float)
        temp2 = mat3[cn][39:52].astype(np.float)
        temp_a = np.array([])
        temp_b = np.array([])
        cnn = 0
        flag = False
        lam =1
        while cnn < 13:
            ter = np.array([float(temp[cnn]),float(temp[cnn+13]),float(temp[cnn+26])])
            av = np.mean(ter)
            std = np.std(ter)
            if cnn ==0:
                sa = av -temp2[cnn]
            vall = temp2[cnn] +sa
            print(av, std)
            temp_a =np.hstack((temp_a,av))
            temp_b =np.hstack((temp_b,std))
            if vall > av +2*std or vall < av - 2*std:
                if vall > av +3*std or vall < av - 3*std:
                    if vall > av +4*std or vall < av - 4*std:
                        lam *=((pouti1/pouti0)*(pouti1/pouti0)*(pouti1/pouti0))
                    else:
                        lam *=((pouti1/pouti0)*(pouti1/pouti0))
                else:
                    lam *=(pouti1/pouti0)
                if lam >= c2 and flag == False:
                    plt.scatter(cnn+1,vall,c="red")
                    flag = True
                elif lam > c1 and lam < c2 and flag == False:
                    plt.scatter(cnn+1, vall,c="blue")
                else:
                    plt.scatter(cnn+1, vall,c="gray")
            else:
                lam *=(pini1/pini0)
                if lam <= 1:
                    lam =1
                    plt.scatter(cnn+1, vall,c="gray")
                else:
                    plt.scatter(cnn+1, vall,c="blue")
            cnn += 1
        if flag == True:
            plt.grid(True)
            plt.hold(True)
            plt.plot(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13]),temp_a)
            print(rf[rf3[cn]])
            plt.savefig(path+"\\figs\\"+rf[rf3[cn]]+".png")
        plt.close()
        cn +=1
print("FINISHED")