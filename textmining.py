#author;R.Kunimoto, TAKENAKA co.
#coding:utf-8

import os
import csv
from janome.tokenizer import Tokenizer
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim



path = "\\\\S0fff001\\9共有フォルダ\\P0統計解析\\部門内\\10_テーマ別\\E1_営業活動\\170627_匿名化サンプル\\kunimoto"
t = Tokenizer(path+"\\janomedic.csv", udic_type="simpledic", udic_enc="shift-jis")

rf = csv.reader(open(path+"\\npwAB_trans.csv","r+"))
wf = csv.writer(open(path+"\\test.csv","w+", newline=""), delimiter=" ")

ff = True
tete = []
for line in rf:
    if ff == True:
        ff = False
        continue
    templ = []
    temp = t.tokenize(line[1])
    for token in temp:
        if '名詞' in token.part_of_speech:
            templ.append(token.base_form)
    wf.writerow(templ)
    tete.append(templ)

print(tete)

titi = []
for ten in tete:
    temtem = ""
    for inko in ten:
        temtem = temtem + inko + " "
    titi.append(temtem)
"""
for li in titi:
    print(li)
"""
count_vectorizer = CountVectorizer(max_features=3000)
tf = count_vectorizer.fit_transform(titi)
features = count_vectorizer.get_feature_names()

# print(features)
wfea = csv.writer(open(path+"\\features.csv","w+", newline=""))
wfea.writerow(features)

tfidf_vect = TfidfVectorizer(max_features=3000, norm='l2')
X = tfidf_vect.fit_transform(titi)

# wf3 = csv.writer(open(path+"\\result.csv","w+", newline=""), delimiter=" ")

numpy.savetxt(path+"\\result.csv",X.toarray(),delimiter=",")

# print(features)

reaf = csv.reader(open(path+"\\result.csv","r+"))

mfcoss = []

for line in reaf:
    mfcoss.append(line)

tfidf_matrix = []

def calc_coss(roww1,roww2):
    r1_norm = 0
    r2_norm = 0
    inn = 0
    t1 = 0
    while t1 < len(roww1):
        r1_norm += float(roww1[t1])**2
        r2_norm += float(roww2[t1])**2
        inn += float(roww1[t1])*float(roww2[t1])
        t1 += 1
    return inn/(numpy.sqrt(r1_norm)*numpy.sqrt(r2_norm))

i1 = 0
while i1 < len(mfcoss):
    i2 = 0
    temp = []
    while i2 < len(mfcoss):
        temp.append(calc_coss(mfcoss[i1], mfcoss[i2]))
        i2 += 1
    tfidf_matrix.append(temp)
    i1 += 1

# print(tfidf_matrix)

tfw = csv.writer(open(path+"\\tfidf_res.csv","w+", newline=""))
for line in tfidf_matrix:
    tfw.writerow(line)

print("end")



"""
print(copus)
copus = numpy.array(copus)
# print(copus)
print(len(copus))
"""