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

titi = []
for ten in tete:
    temtem = ""
    for inko in ten:
        temtem = temtem + inko + " "
    titi.append(temtem)


dictionary = gensim.corpora.Dictionary(tete)
# print(dictionary.token2id)

# dictionary.filter_extremes(no_below=20, no_above=0.3)
# no_berow: 使われてる文章がno_berow個以下の単語無視
# no_above: 使われてる文章の割合がno_above以上の場合無視

# dictionary.save_as_text('livedoordic.txt')

# dictionary = corpora.Dictionary.load_from_text('livedoordic.txt')
"""
tmp = dictionary.doc2bow(tete)
dense = list(gensim.matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
print(dense)
"""
tete.insert(0,str(2))
"""
unko = csv.writer(open(path+"\\ldates\\goal.csv","w+", newline=""),delimiter = " ")
for line in titi:
    unko.writerow(str(line))
"""

corpus = gensim.corpora.lowcorpus.LowCorpus(path + "\\ldates\\dataen2.txt")
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=25, id2word=corpus.id2word)

print("end")



"""
print(copus)
copus = numpy.array(copus)
# print(copus)
print(len(copus))
"""