#author;R.Kunimoto, TAKENAKA co.
#coding:utf-8

import os
import csv
import numpy as np
import pandas as pd
import sys
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import PreTrainingChain
import re
import matplotlib as plt

n_hl = 2
n_hlu = 100
n_input = 3
n_output = 2

# units_row = [3,100,100,2]
# n_input = 3
# n_l2 = 100
# n_out = 2

path = os.getcwd()
f = open(path+"\\testmod.py", "w+", newline="")

program_description = """\n
"""
program_description += """class MLP(Chain):
\tdef __init__(self):
\t\tsuper(MLP, self).__init__(\n"""


# chainer.optimizers.
optlist = [
    "AdaDelta",
    "AdaGrad",
    "Adam",
    "MomentumSGD",
    "NesterovAG",
    "RMSprop",
    "RMSpropGraves",
    "SGD",
    "SMORMS3"]
optn = 0
optim = optlist[optn]

i = 1
while i <= n_hl+1:
    if i == 1:
        program_description += "\t\t\t"+"l"+str(i)+" = L.Linear(n_input, n_hlu),\n"
    elif i == n_hl+1:
        program_description += "\t\t\t"+"l"+str(i)+" = L.Linear(n_hlu, n_output),\n"
    else:
        program_description += "\t\t\t"+"l"+str(i)+" = L.Linear(n_hlu, n_hlu),\n"
    i += 1

program_description += "\t\t)\n"

program_description += """\tdef __call__(self, x):\n"""

j = 1
while j <= n_hl+1:
    if j == 1:
        program_description += "\t\t"+"h"+str(j)+" = F.relu(self.l"+str(j)+"(x))\n"
    elif j == n_hl+1:
        program_description += "\t\t"+"y = self.l"+str(j)+"(h"+str(j-1)+")\n\t\treturn y\n"
    else:
        program_description += "\t\t"+"h"+str(j)+" = F.relu(self.l"+str(j)+"(h"+str(j-1)+"))\n"
    j += 1

for line in program_description:
    f.write(line)

f.close()

print(program_description)

logfilename = "dnnlog.txt"

# sys.stdout = open(path+"tmep.txt","w+")
# csvファイルの読み込み
data_f = pd.read_csv(path+'\\datasets\\titanic.csv', header=0)

# 関係ありそうなPClass,Sex,Ageのみを使う
data_f = data_f[["Pclass", "Sex", "Age", "Survived"]]

# Ageの欠損値を中央値で補完
data_f["Age"] = data_f["Age"].fillna(data_f["Age"].median())
# maleは1, femaleは0に置換
data_f["Sex"] = data_f["Sex"].replace("male", 1)
data_f["Sex"] = data_f["Sex"].replace("female", 0)

data_array = data_f.as_matrix()

X = []
Y = []
for x in data_array:
    x_split = np.hsplit(x, [3,4])
    X.append(x_split[0].astype(np.float32))
    Y.append(x_split[1].astype(np.int32))

X = np.array(X)
Y = np.ndarray.flatten(np.array(Y))

# 891個のデータのうち623個(7割)を訓練用データ、残りをテスト用データにする
train, test = datasets.split_dataset_random(datasets.TupleDataset(X, Y), 623)
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

exec(program_description)

model = L.Classifier(MLP())
exec("optimizer = optimizers."+optim+"()\n"
     "optimizer.setup(model)")

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
print(model.predictor())
