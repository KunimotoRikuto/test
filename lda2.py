#author;R.Kunimoto, TAKENAKA co.
#coding:utf-8

import os
import csv
from janome.tokenizer import Tokenizer
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim

path = "\\\\S0fff001\\9共有フォルダ\\P0統計解析\\部門内\\10_テーマ別\\E1_営業活動\\170627_匿名化サンプル\\kunimoto"
"""
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

titi = []
for ten in tete:
    temtem = ""
    for inko in ten:
        temtem = temtem + inko + " "
    titi.append(temtem)

dictionary = gensim.corpora.Dictionary(tete)
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000)  # 変更
corpus = [dictionary.doc2bow(text) for text in tete]
"""
corpus = gensim.corpora.lowcorpus.LowCorpus(path + '\\ldates\\dataen2.txt')
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=5, id2word=corpus.id2word)

for topic in lda.show_topics(-1):
    print(topic)

for topics_per_document in lda[corpus]:
    print(topics_per_document[0][0])