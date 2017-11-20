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

goal = 12
models = ["SVC","RFC", "kNNC"]
# models = ["SVC","RFC"]
shurui_array = ["rebar","iron","coal","scrap","oil","bdi","ypd","elec","const"]
shurui_array = ["rebar","iron","coal","scrap","oil","bdi"]
shurui_array = ["rebar","iron","coal","scrap","oil","bdi","elec"]

karaterm = [3,6,12]

kyoshi=[]
# 高々9じゃないと不定方程式化するよ！
term_m = 9
array_ratio = [0, 0.03, 0.05, 0.1]

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
n_data = len(df)-1
print("ndata",n_data)

def labeler(ratio, orig, val, sp):
    und = 0 - orig[sp]*ratio
    upp = 0 + orig[sp]*ratio
    if val[sp]-orig[sp] < und:
        return -1
    elif val[sp]-orig[sp] >= und and val[sp]-orig[sp] <= upp:
        return 0
    else:
        return 1

def evaler(act,pre,term,ist,val):
    iii = 0
    maru = 0.0
    batsu = 0.0
    if ist == True:
        while iii < term:
            if act[iii]==val and act[iii]==pre[iii]:
                maru += 1
            elif act[iii]==val and act[iii]!=pre[iii]:
                batsu += 1
            iii += 1
        if maru+batsu != 0:
            return maru / (maru+batsu)
        else:
            return "null"
    elif ist == False:
        while iii < term:
            if pre[iii]==val and act[iii]==pre[iii]:
                maru += 1
            elif pre[iii]==val and act[iii]!=pre[iii]:
                batsu += 1
            iii += 1
        if maru+batsu != 0:
            return maru / (maru+batsu)
        else:
            return "null"

def eval_all(act,pre,termm):
    iiii = 0
    maru = 0.0
    batsu = 0.0
    while iiii < termm:
        if act[iiii] == pre[iiii]:
            maru += 1
        else:
            batsu += 1
        iiii += 1
    return maru / (maru+batsu)


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
                # 訂正 k+j+1 -> k+j
                tmp = df[k+j] + tmp
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
                    if t != n_data-goal*2+2:
                        if ss == 0:
                            temp.append(float(labeler(r, df[t-1], df[t], s)))
                        else:
                            temp.append(float(labeler(r, df[t], df[t+ss], s)))
                    else:
                        if ss == 0:
                            temp.append(float(labeler(r, df[t+goal-2], df[t+goal-1], s)))
                        else:
                            temp.append(float(labeler(r, df[t+goal-1], df[t+ss+goal-1], s)))
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
        pred_row = []
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
                        # clf = svm.SVC(kernel='linear')
                        clf = svm.SVC()
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
            # print(len(svctmp),len(rfctmp),len(knnctmp))
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
print(len(testing_goal))
print(len(testing_goal[0]))
print(len(testing_goal[0][0]))
print(len(testing_goal[0][0][0]))

print(len(pred_row_temp))
print(len(pred_row_temp[0]))
print(len(pred_row_temp[0][0]))
print(len(pred_row_temp[0][0][0]))
print(len(pred_row_temp[0][0][0][0]))
print(len(pred_row_temp[0][0][0][1]))
print(len(pred_row_temp[0][0][0][2]))




tinko = 0
while tinko < term_m:
    wdata.writerow(pred_row_temp[tinko][0][0][0])
    tinko += 1
"""
wdata.writerow(["Sp.","Ratio","goalterm","Cat.","Val.","Score","LearnTerm","Alg."])
# print("kyoshi",kyoshi)
shu = 0
while shu < len(shurui_array):
    os.mkdir(path+"\\"+shurui_array[shu])
    roo = 0
    while roo < len(array_ratio):
        os.mkdir(path+"\\"+shurui_array[shu]+"\\ratio_"+str(array_ratio[roo]))
        mod = 0
        chon = []
        for inn in testing_goal[roo]:
            chon.append(inn[shu])
        while mod < len(models):
            ep = 1
            while ep <= term_m:
                print(pred_row_temp[ep-1][shu][roo][mod])
                plt.subplot(3,3,ep)
                plt.legend()
                plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11])
                plt.yticks([-1,0,1])
                plt.ylim([-1,1])
                plt.plot(numpy.array([0,1,2,3,4,5,6,7,8,9,10,11]), numpy.array(pred_row_temp[ep-1][shu][roo][mod]), label=models[mod], marker=".", alpha=0.5)
                # plt.plot(x,y)
                plt.tick_params(labelsize=7)
                plt.title("term_"+str(ep), fontsize=7)
                plt.plot(numpy.array([0,1,2,3,4,5,6,7,8,9,10,11]), numpy.array(chon), label="act", marker=".", alpha=0.5)
                for pin in karaterm:
                    valval = -1
                    wdata.writerow([shurui_array[shu],array_ratio[roo],str(pin),"Accuracy","null", str(eval_all(chon,pred_row_temp[ep-1][shu][roo][mod],pin)),str(ep),models[mod]])
                    while valval <= 1:
                        wdata.writerow([shurui_array[shu],array_ratio[roo],str(pin),"Precision",str(valval),str(evaler(chon,pred_row_temp[ep-1][shu][roo][mod],pin,True,valval)),str(ep),models[mod]])
                        wdata.writerow([shurui_array[shu],array_ratio[roo],str(pin),"Recall",str(valval),str(evaler(chon,pred_row_temp[ep-1][shu][roo][mod],pin,False,valval)),str(ep),models[mod]])
                        valval += 1
                ep += 1
            os.mkdir(path+"\\"+shurui_array[shu]+"\\ratio_"+str(array_ratio[roo])+"\\"+models[mod])
            plt.savefig(path+"\\"+shurui_array[shu]+"\\ratio_"+str(array_ratio[roo])+"\\"+models[mod]+"\\"+shurui_array[shu]+"_ratio_"+str(array_ratio[roo])+".png")
            plt.close()
            mod += 1
        print("saved")
        roo += 1
    shu += 1
    # wdata.writerow(tra_x)

print("end")
