#author;R.Kunimoto, Informa Engineering Div, TAKENAKA co.
#coding:utf-8

import csv
import re
import numpy
import chainer
import pandas
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.functions.evaluation.accuracy import accuracy

r_filepath = "C:\\Users\\1500570\\Documents\\R\\WS\\train_iron3.csv"
data = pandas.read_csv(r_filepath)
data.columns = ["rebar","iron","coal","scrap","oil","bdi","ypd","elec","const","missi",
                "rebar2","iron2","coal2","scrap2","oil2","bdi2","ypd2","elec2","const2","missi2",
                "rebar3","iron3","coal3","scrap3","oil3","bdi3","ypd3","elec3","const3","missi3",
                "t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12"]
data = data.T

x_train = data[0:len(data)-12].T
y_train = data[len(data)-12:len(data)].T

x_test = x_train
y_test = y_train

x = chainer.Variable(x_train.values.astype(numpy.float32), volatile=False)

print(x)

x_train = x_train.values
x_test = x_test.values

train_iter = iterators.SerialIterator(x_train, batch_size=10, shuffle=True)
test_iter = iterators.SerialIterator(x_test, batch_size=200, repeat=False, shuffle=False)

class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            l1 = L.Linear(None,n_units),
            l2 = L.Linear(None,n_units),
            l3 = L.Linear(None,n_out),
            )
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({"loss":loss, "accuracy":accuracy}, self)
        return loss

model = Classifier(MLP(10,1))
optimizer = optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out = 'result')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()
"""
batchsize = 10
datasize = len(x_train)

for epoch in range(20):
    print("epoch %d" % epoch)
    indexes = numpy.random.permutation(datasize)
    for i in range(0, datasize, batchsize):
        x = Variable(x_train[indexes[i:i+batchsize]])
        t = Variable(y_train[indexes[i:i+batchsize]])
        optimizer.updata(model, x, t)
"""
print("Program finished")