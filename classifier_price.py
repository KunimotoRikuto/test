#coding:utf-8

from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas
import numpy
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sympy.interactive.session import preexec_source
import csv
import matplotlib.pyplot as plt
import os

# ハイパラ、基本設定群
path = "C:\\Users\\1500570\\Documents\\R\\WS\\dataset_3class_stayzone"
data = csv.reader(open(path+"\\data1.csv","r"))
wdata = csv.writer(open(path+"\\result1.csv","w+", newline=""))
# wdata = csv.writer(open(path+"\\resulet.csv","w+",newline=""))
df = []
n_data = 131
goal = 12
models = ["SVC","RFC", "kNNC"]
# models = ["SVC","RFC"]
pred_row = []
shurui_array = ["rebar","iron","coal","scrap","oil","bdi","ypd","elec","const"]
shurui_array = ["rebar","iron","coal","scrap","oil","bdi"]
kyoshi=[]
# 高々9じゃないと不定方程式化するよ！
term_m = 9
array_ratio = [0, 0.03, 0.05, 0.1]
# array_ratio = [0, 0.03]

fff = True
for line in data:
    if fff == True:
        fff = False
        df.append(line)
        continue
    else:
        tmtm = []
        for line2 in line:
            tmtm.append(float(line2))
        df.append(tmtm)
first = df[1]
print(first)

def labeler(ratio, val, sp):
    global first
    und = 0 - first[sp]*ratio
    upp = 0 + first[sp]*ratio
    if val < und:
        return -1
    elif val >= und and val <= upp:
        return 0
    else:
        return 1

# 最初のデータは前回データがなくクラス定義できないのでスキップ
i = 1
pred_row_temp = []
while i <= term_m:
    tra_x = []
    tes_x = []
    k = 1
    while k <= n_data-i-goal*2+2:
        j = 0
        tmp = []
        while j < i:
            if k != n_data-i-goal*2+2:
                tmp = df[k+j+1] + tmp
            else:
                tmp = df[k+j+goal] + tmp
            j += 1
        if k != n_data-i-goal*2+2:
            tra_x.append(tmp)
        else:
            tes_x.append(tmp)
        k += 1
    # print("trax",tra_x)
    tra_y = []
    tes_y = []
    for r in array_ratio:
        t = i+1
        temp3 = []
        tempp3 = []
        while t <= n_data-goal*2+2:
            temp2 = []
            tempp2 = []
            ss = 0
            while ss < goal:
                temp = []
                s = 0
                while s < len(shurui_array):
                    if t!= n_data-goal*2+2:
                        temp.append(float(labeler(r, df[t+ss][s]-df[t+ss-1][s], s)))
                    else:
                        temp.append(float(labeler(r, df[t+ss+goal-1][s]-df[t+ss+goal-2][s], s)))
                    s += 1
                ss += 1
                if t != n_data-goal*2+2:
                    temp2.append(temp)
                else:
                    tempp2.append(temp)
            if t != n_data-goal*2+2:
                temp3.append(temp2)
            else:
                tempp3 = tempp2
            t += 1
        tra_y.append(temp3)
        tes_y.append(tempp3)
    # print("tra_x",tra_x[:3])
    # print("tra_y",tra_y[1][:3])
    training_exp = numpy.array(tra_x)
    training_goal = numpy.array(tra_y)
    testing_exp = numpy.array(tes_x)
    testing_goal = numpy.array(tes_y)
    # print("testing goal",testing_goal)
    # print("testing goal",testing_goal[0])
    # print("traexlen",len(training_exp))
    # print("traing exp",training_exp)
    kyoshi.append(testing_goal)
    # print("tes_x",testing_exp)
    # print("tes_y",testing_goal)
    #print(len(tra_y[0]))
    #print(len(tra_x))
    #print("tra_x",training_exp)
    #print("tra_y",training_goal[0])
    # print(len(tra_x))

    shurui = 0
    rrrarray = []
    while shurui < len(shurui_array):
        rrr = 0
        while rrr < len(array_ratio):
            aite = 0
            svctmp = []
            rfctmp = []
            knnctmp = []
            while aite < goal:
                ty = []
                for unko in training_goal[rrr]:
                    # print(unko[aite][shurui])
                    ty.append(unko[aite][shurui])
                array_unko = numpy.array(ty)
                ep = 0
                while ep < len(models):
                    if ep == 0:
                        print("i=",i,",shurui=",shurui,",rrr=",rrr,",aite=",aite,", SVC is starting")
                        clf = svm.SVC(kernel='linear')
                        # print(len(training_exp))
                        # print(array_unko)
                        clf.fit(training_exp,array_unko)
                        svctmp.append(float(clf.predict(testing_exp)))
                    elif ep == 1:
                        # print("i=",i,",shurui=",shurui,",rrr=",rrr,",aite=",aite,", RFC is starting")
                        clf = RandomForestClassifier()
                        clf.fit(training_exp,array_unko)
                        rfctmp.append(float(clf.predict(testing_exp)))
                    elif ep == 2:
                        # print("i=",i,",shurui=",shurui,",rrr=",rrr,",aite=",aite,", kNNC is starting")
                        clf = neighbors.KNeighborsClassifier()
                        clf.fit(training_exp,array_unko)
                        knnctmp.append(float(clf.predict(testing_exp)))
                    ep += 1
                aite += 1
            rrr += 1
            # 各識別問題別にterm0-11の回帰結果リストがopされるので親元に追加
            pred_row.append([svctmp,rfctmp,knnctmp])
            # print("pred_row",pred_row)
        rrrarray.append(pred_row)
        # print("rrrarray",rrrarray)
        shurui += 1
    pred_row_temp.append(rrrarray)
    # print(pred_row_temp)
    i += 1
    # wdata.writerow(tes_x)

# print(len(training_exp))
# print(len(training_exp[2]))
"""
print(len(training_goal))
print(len(training_goal[0]))
print(len(training_goal[0][0]))
print(len(training_goal[0][0][0]))

tinko = 0
while tinko < term_m:
    wdata.writerow(pred_row_temp[tinko][0][0][0])
    tinko += 1
"""
print("kyoshi",kyoshi)
shu = 0
while shu < len(shurui_array):
    os.mkdir(path+"\\"+shurui_array[shu])
    roo = 0
    while roo < len(array_ratio):
        os.mkdir(path+"\\"+shurui_array[shu]+"\\_ratio_"+str(array_ratio[roo]))
        ep = 1
        chon = []
        for inn in testing_goal[roo]:
            chon.append(inn[shu])
        while ep <= term_m:
            mod = 0
            plt.subplot(3,3,ep)
            plt.legend()
            plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11])
            plt.yticks([-1,0,1])
            while mod < len(models):
                print(pred_row_temp[ep-1][shu][roo][mod])
                plt.plot(numpy.array([0,1,2,3,4,5,6,7,8,9,10,11]), numpy.array(pred_row_temp[ep-1][shu][roo][mod]), label=models[mod], marker=".", alpha=0.5)
                # plt.plot(x,y)
                plt.tick_params(labelsize=7)
                mod += 1
            plt.title("term_"+str(ep), fontsize=7)
            ep += 1
            plt.plot(numpy.array([0,1,2,3,4,5,6,7,8,9,10,11]), numpy.array(chon), label="act", marker=".", alpha=0.5)
        plt.savefig(path+"\\"+shurui_array[shu]+"\\_ratio_"+str(array_ratio[roo])+"\\_shu_"+shurui_array[shu]+"_ratio_"+str(array_ratio[roo])+".png")
        plt.close()
        print("saved")
        roo += 1
    shu += 1
    # wdata.writerow(tra_x)

print("end")
