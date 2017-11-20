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

units_row = [3,100,100,2]
n_input = 3
n_l2 = 100
n_out = 2

path = os.getcwd()
f = open(path+"\\testmod.py", "w+", newline="")

program_description = """import numpy as np
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
    def __init__(self):
        super(MLP, self).__init__(\n"""

i = 1
while i <= len(units_row):
    program_description += "\t\t\t"+"l"+str(i)+" = L.Linear(units_row["+str(i)+"], units_row["+str(i+1)+"])\n"
    i += 1

program_description += "\t\t)\n"

program_description += """\tdef __init__(self):\n"""

j = 1
while j <= len(units_row):
    if j == 1:
        program_description += "\t\t"+"h"+str(j)+" = F.relu(self.l"+str(j)+"(x)\n"
    elif j == len(units_row):
        program_description += "\t\t"+"y = self.l"+str(j)+"(h"+str(j-1)+")\n\t\treturn y\n"
    else:
        program_description += "\t\t"+"h"+str(j)+" = F.relu(self.l"+str(j)+"(h"+str(j-1)+"))\n"
    j += 1

for line in program_description:
    f.write(line)

f.close()

print(program_description)