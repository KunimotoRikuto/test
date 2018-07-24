#author;R.Kunimoto, TAKENAKA co.
#coding:utf-8

import math
import os
import csv
import numpy as np
import pandas as pd
import sys
from chainer import training
from chainer import report, datasets, iterators, optimizers, serializers, Variable
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import PreTrainingChain
import re
import matplotlib as plt
from chainer.functions.loss.mean_squared_error import mean_squared_error as mse
from sklearn.model_selection import train_test_split

over_count =0
units_row = [19,100,19,1]
# n_input = 3
# n_l2 = 100
# n_out = 2

# resの番号を決める大事なやつ
goal_num =7

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
program_description += "\tdef pred(self, x):\n"

j = 1
while j <= len(units_row)-1:
    if j == 1:
        program_description += "\t\t"+"h"+str(j)+" = F.maxout(self.l"+str(j)+"(x),1)\n"
    elif j == len(units_row)-1:
        program_description += "\t\t"+"y = self.l"+str(j)+"(h"+str(j-1)+")\n\t\treturn y\n"
    else:
        program_description += "\t\t"+"h"+str(j)+" = F.maxout(self.l"+str(j)+"(h"+str(j-1)+"),1)\n"
    j += 1

for line in program_description:
    f.write(line)

f.close()

print(program_description)

logfilename = "dnnlog.txt"

# sys.stdout = open(path+"tmep.txt","w+")
# csvファイルの読み込み
data_f = pd.read_csv("C:\\Users\\1500570\\Documents\\R\\WS\\ds_dr\\"+"180122_sdata.enc.train2.csv", header=0)
data_honban =pd.read_csv("C:\\Users\\1500570\\Documents\\R\\WS\\ds_dr\\"+"180122_sdata.enc.test.Var2.csv", header=0)

# data_f = pd.read_csv(path+'\\datasets\\titanic.csv', header=0)


fieldarr = ["Var01_f","Var02_f","Var03_f","Var04_f","Var05_f","Var06_f","Var07_f","Var08_f","Var09_n","Var10_n",
            "Var11_n","Var12_n","Var13_n","Var14_f","Var15_f","Var16_f","Var17_n","Var18_n","Var19_n",
            "Res01_n","Res02_n","Res03_n","Res04_n","Res05_n","Res06_n","Res07_n","Res08_f"]
fieldarr2 = ["Var01_f","Var02_f","Var03_f","Var04_f","Var05_f","Var06_f","Var07_f","Var08_f","Var09_n","Var10_n",
            "Var11_n","Var12_n","Var13_n","Var14_f","Var15_f","Var16_f","Var17_n","Var18_n","Var19_n"]
# 使う変数の選択（目的、説明コミ）
data_f = data_f[fieldarr]
data_honban =data_honban[fieldarr2]


# 欠損値補完
i = 0
for line in fieldarr:
    data_f[line] =data_f[line].fillna(data_f[line].median())

for line2 in fieldarr2:
    data_honban[line2] =data_honban[line2].fillna(data_honban[line2].median())

# maleは1, femaleは0に置換
# data_f["Sex"] = data_f["Sex"].replace("male", 1)
# data_f["Sex"] = data_f["Sex"].replace("female", 0)

data_array = np.hsplit(data_f.as_matrix(),[19])
data_array2 = np.hsplit(data_honban.as_matrix(),[19])

X = data_array[0].astype(np.float32)
Y = data_array[1].T[goal_num].astype(np.float32)
Xhonban =data_array2[0].astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, Y, train_size=0.8, random_state=1)

# 891個のデータのうち623個(7割)を訓練用データ、残りをテスト用データにする
# train, test = datasets.split_dataset_random(datasets.TupleDataset(X, Y), 672)
# train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
# test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

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


optimizer = optimizers.AdaGrad()
optimizer.setup(model)

loss_val = 100
loss_val2 =100
epoch = 0
temp_model =[]
while loss_val > 1e-5:
    x = Variable(X_train.reshape(672,19))
    t = Variable(y_train.reshape(672,1))
    model.zerograds()
    y = model.pred(x)
    loss = mse(y,t)
    loss.backward()
    optimizer.update()
    if epoch % 1000 == 0:
        temp_model.append(model)
        xx =Variable(X_val.reshape(168,19))
        tt =Variable(y_val.reshape(168,1))
        yy =model.pred(xx)
        loss2 =mse(yy,tt)
        loss_val = loss.data
        temp = loss_val2
        loss_val2 =loss2.data
        if temp < loss_val2:
            over_count +=1
        else:
            over_count =0
        print('epoch:', epoch)
        print('train mean loss ={}'.format(math.sqrt(loss_val)))
        print('test mean loss ={}'.format(math.sqrt(loss_val2)))
        print(' - - - - - - - - - ')
    if epoch >= 100000 or over_count>=2:
        break
    epoch += 1

fout =open("C:\\Users\\1500570\\Documents\\R\\WS\\ds_dr\\Res0"+str(goal_num+1)+"_n_alldata.csv", "w+", newline="")
datawriter = csv.writer(fout)
fout2 =open("C:\\Users\\1500570\\Documents\\R\\WS\\ds_dr\\Res0"+str(goal_num+1)+"_n_simdata.csv", "w+", newline="")
datawriter2 = csv.writer(fout2)
fout3 =open("C:\\Users\\1500570\\Documents\\R\\WS\\ds_dr\\res0"+str(goal_num+1)+"_n_testdata.csv", "w+", newline="")
datawriter3 = csv.writer(fout3)

summ_x =Variable(X.reshape(840,19))
summ_y =temp_model[epoch%1000 -2].pred(summ_x)
summ_x2 =Variable(Xhonban.reshape(209,19))
summ_y2 =temp_model[epoch%1000 -2].pred(summ_x2)
summ_x3 =Variable(X_val.reshape(168,19))
summ_y3 =temp_model[epoch%1000 -2].pred(summ_x3)

for line in summ_y:
    datawriter.writerow(line.data)
fout.close()

for line2 in summ_y2:
    datawriter2.writerow(line2.data)
fout2.close()

for line3 in summ_y3:
    datawriter3.writerow(line3.data)

for i in y_val:
    print(i)

"""
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
"""
print("unko")
# print(model.predictor())
