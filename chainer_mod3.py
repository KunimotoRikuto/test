import numpy as np
import pandas as pd
import chainer
import sys
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import PreTrainingChain

n_input = 3
n_l2 = 100
n_out = 2
size_l = 1
m = open("chainer_sub1.py")

path = 'C:\\Users\\1500570\\Documents\\R\\WS\\'
# sys.stdout = open(path+"tmep.txt","w+")
# csvファイルの読み込み
data_f = pd.read_csv(path+'titanic.csv', header=0)

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

class MLP(Chain):
    def __init__(self, layer_sizes=[3,100,2]):
        self.size = len(layer_sizes)
        super(MLP, self).__init__(
            exec(m)
            )

    def __call__(self, x):
        h0 = F.relu(self.l0(x))
        for idx in range(self.size-1):
            exec("        h%d = F.relu(self.l%d(h%d))" % (idx+1, id+1, idx))
        exec("        y = self.l%d(h%d)" % (self.size, self.size-1))
        return y

model = L.Classifier(MLP())
optimizer = optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (30, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport(log_name="rikutolog2.txt"))
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()
print("unko")