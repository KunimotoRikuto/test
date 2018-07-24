#author;R.Kunimoto, TAKENAKA co.
#coding:utf-8

import os
import csv
import numpy as np
import pandas as pd
import sys
from chainer import training
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import PreTrainingChain
import re
import matplotlib as plt
from chainer import Variable


units_row = [19,100,100,1]
# n_input = 3
# n_l2 = 100
# n_out = 2

goal_num =0

path = os.getcwd()
f = open(path+"\\testmod.py", "w+", newline="")

program_description = """
"""

iij = 0
while iij < len(units_row):
    if iij == 0:
        program_description += "units_row = ["+str(units_row[iij])+","
    elif iij == len(units_row)-1:
        program_description += str(units_row[iij])+"]\n"
    else:
        program_description += str(units_row[iij])+","
    iij += 1

program_description += """class MLP(Chain):
\tdef __init__(self):
\t\tsuper(MLP, self).__init__(\n"""

i = 1
while i <= len(units_row)-1:
    if i == 0 or i == len(units_row)-1:
        program_description += "\t\t\t"+"l"+str(i)+" = L.Linear(units_row["+str(i-1)+"], units_row["+str(i)+"]),\n"
    else:
        program_description += "\t\t\t"+"l"+str(i)+" = L.Linear(units_row["+str(i-1)+"], units_row["+str(i)+"]),\n"
    i += 1

program_description += "\t\t)\n"
program_description += "\tdef __call__(self, x):\n"

j = 1
while j <= len(units_row)-1:
    if j == 1:
        program_description += "\t\t"+"h"+str(j)+" = F.sigmoid(self.l"+str(j)+"(x))\n"
    elif j == len(units_row)-1:
        program_description += "\t\t"+"y = self.l"+str(j)+"(h"+str(j-1)+")\n\t\treturn y\n"
    else:
        program_description += "\t\t"+"h"+str(j)+" = F.sigmoid(self.l"+str(j)+"(h"+str(j-1)+"))\n"
    j += 1

for line in program_description:
    f.write(line)

f.close()

print(program_description)

logfilename = "dnnlog.txt"

# sys.stdout = open(path+"tmep.txt","w+")
# csvファイルの読み込み
data_f = pd.read_csv("C:\\Users\\1500570\\Documents\\R\\WS\\ds_dr\\"+"180122_sdata.enc.train2.csv", header=0)
# data_f = pd.read_csv(path+'\\datasets\\titanic.csv', header=0)


fieldarr = ["Var01_f","Var02_f","Var03_f","Var04_f","Var05_f","Var06_f","Var07_f","Var08_f","Var09_n","Var10_n",
            "Var11_n","Var12_n","Var13_n","Var14_f","Var15_f","Var16_f","Var17_n","Var18_n","Var19_n",
            "Res01_n","Res02_n","Res03_n","Res04_n","Res05_n","Res06_n","Res07_n","Res08_f"]
# 使う変数の選択（目的、説明コミ）
data_f = data_f[fieldarr]

# 欠損値補完
i = 0
for line in fieldarr:
    print(line)
    data_f[line] =data_f[line].fillna(data_f[line].median())

# maleは1, femaleは0に置換
# data_f["Sex"] = data_f["Sex"].replace("male", 1)
# data_f["Sex"] = data_f["Sex"].replace("female", 0)

data_array = data_f.as_matrix()

X = []
Y = []
for x in data_array:
    x_split = np.hsplit(x, [19,20])
    X.append(x_split[0].astype(np.float32))
    Y.append(x_split[1][goal_num].astype(np.float32))

X = Variable(np.array(X))
Y = Variable(np.ndarray.flatten(np.array(Y)))


print(len(X))
print(len(Y))

# 891個のデータのうち623個(7割)を訓練用データ、残りをテスト用データにする
train, test = datasets.split_dataset_random(datasets.TupleDataset(X, Y), 672)
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

"""
class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(n_input, n_l2),
            l2=L.Linear(n_l2, n_l2),
            l3=L.Linear(n_l2, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y
"""

exec(program_description)

model = MLP()
optimizer = optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (30, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport(log_name = path+logfilename))
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()

f = open(path+logfilename, "r+")
seiki_ma = re.compile('validation/main/accuracy.:.0.\d+')
seiki_ma2 = re.compile('0.\d+')

y_line = []
for line in f:
    try:
        ttmm = seiki_ma2.search(seiki_ma.search(line).group()).group()
        print(ttmm)
        y_line.append(ttmm)
    except:
        continue

x_line = np.arange(0,len(y_line),1)
plt.pyplot.plot(x_line, y_line)
plt.pyplot.show()

print("unko")
# print(model.predictor())
