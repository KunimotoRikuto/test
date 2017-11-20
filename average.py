#author;R.Kunimoto, TAKENAKA co.
#coding:utf-8
import csv
import os

speci = "const"
month = 12
path ="C:\\Users\\1500570\\Documents\\R\\WS\\result_"+speci+"\\"+str(month)+"\\"

listave = []
for line in os.listdir(path):
    f = open(path+line,"r")
    ff = True
    for i in csv.reader(f):
        if ff == False:
            listave.append(float(i[0]))
        else:
            ff = False

print(sum(listave)/len(listave))
print("unko")