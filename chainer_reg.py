#author;R.Kunimoto, Informa Engineering Div, TAKENAKA co.
#coding:utf-8

import csv
import re
import json, sys, glob, datetime, math
import numpy
import matplotlib.pyplot as plt
import pandas
import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

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

print(len(x_train))
print(len(y_train))


class RegressionFNN:
    def __init__(self, n_units=64, batchsize=10):
        self.n_units = n_units
        self.batchsize = batchsize
        self.plotcount=0

    def load(self, train_x, train_y):
        if len(train_x)!=len(train_y):
            raise ValueError
        self.N = len(train_y)
        self.I = 30
        self.x_train = numpy.array(train_x, numpy.float32).reshape(len(train_x),1)
        self.y_train = numpy.array(train_y, numpy.float32).reshape(len(train_y),)
        #
        self.model = chainer.FunctionSet(l1=F.Linear(self.I, self.n_units),
                                         l2=F.Linear(self.n_units, self.n_units),
                                         l3=F.Linear(self.n_units, self.n_units),
                                         l4=F.Linear(self.n_units, 12))
        #
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(y_data)
        h1 = F.relu(self.model.l1(x))
        h2 = F.relu(self.model.l2(h1))
        h3 = F.relu(self.model.l3(h2))
        y = F.reshape(self.model.l4(h3), (len(y_data), ))
        return F.mean_squared_error(y, t), y

    def calc(self, n_epoch):
        for epoch in range(n_epoch):
            perm = numpy.random.permutation(self.N)
            sum_loss = 0
            #
            for i in range(0, self.N, self.batchsize):
                x_batch = self.x_train[perm[i:i + self.batchsize]]
                y_batch = self.y_train[perm[i:i + self.batchsize]]
                #
                self.optimizer.zero_grads()
                loss, y = self.forward(x_batch, y_batch)
                loss.backward()
                self.optimizer.update()
                #
                sum_loss += float(loss.data) * len(y_batch)
            #
            print('epoch = {}, train mean loss={}\r'.format(epoch, sum_loss / self.N), end="")

    def getY(self, test_x, test_y):
        if len(test_x)!=len(test_y):
            raise ValueError
        x_test = numpy.array(test_x, numpy.float32).reshape(len(test_x),1)
        y_test = numpy.array(test_y, numpy.float32).reshape(len(test_y),)
        loss, y = self.forward(x_test, y_test, train=False)
        #
        with open("output/{}.csv".format(self.plotcount), "w") as f:
            f.write("\n".join(["{},{}".format(ux,uy) for ux,uy in zip(y.data, y_test)]))

        #
        plt.clf()
        plt.plot(x_test, y_test, color="b", label="original")
        plt.plot(x_test, y.data, color="g", label="RegFNN")
        plt.legend(loc = 'lower left')
        plt.ylim(-1.2,1.2)
        plt.grid(True)
        plt.savefig("output/{}.png".format(self.plotcount))
        self.plotcount += 1


def f(d):
    return [math.sin(u) for u in d]
if __name__=="__main__":
    rf = RegressionFNN(n_units=64, batchsize=10)
    d = [u*0.02 for u in range(314)]
    rf.load(d,f(d))
    rf.calc(1000) # epoch
    rf.getY(d,f(d))
    print("\ndone.")


