#coding:utf-8

from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas
import numpy
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sympy.interactive.session import preexec_source
import csv

typer = ["rebar","bdi","coal","const","elec","iron","missi","oil","scrap","ypd"]
term = 3
f = open("C:\\Users\\1500570\\Documents\\R\\WS\\result0413.csv", "w+", newline = "")
ff = csv.writer(f)

while term <= 24:
    for typerr in typer:
        goal_column = 0

        path = 'C:\\Users\\1500570\\Documents\\R\\WS\\dataset\\'+str(typerr)+'\\'
        pred_row = []
        real_row = []

        data_train = numpy.loadtxt(path+"logit_train_"+str(term)+".csv", delimiter=",")
        data_test= numpy.loadtxt(path+"logit_test_"+str(term)+".csv", delimiter=",")

        while goal_column < 12:
            exp = data_train[:, 0:len(data_train[0])-12]
            goal = data_train[:, len(data_train[0])-12:len(data_train[0])][:,goal_column:goal_column+1]
            preexp = data_test[0:len(data_test)-12]
            pregoal = data_test[len(data_test)-12:len(data_test)][goal_column]
            clf = svm.SVC(kernel='linear', gamma=0.01)
            # clf = RandomForestClassifier()
            clf.fit(exp, goal)
            pred_row.append(int(clf.predict(preexp)))
            real_row.append(int(pregoal))
            goal_column += 1
        print(typerr)
        print(pred_row)
        print(real_row)
        ff.writerows(str(typerr)+"term="+str(term))
        ff.writerow(pred_row)
        ff.writerow(real_row)
    term += 3
f.close()

"""
print(classification_report(pregoal, test_pred))
print(accuracy_score(pregoal, test_pred))
"""