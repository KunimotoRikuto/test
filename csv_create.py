#author;R.Kunimoto, TAKENAKA co.
#coding:utf-8
import csv

path = "C:\\Users\\1500570\\Documents\\R\\WS"
fn = "data1.csv"
r_filepath = path + "\\" + fn
fr = open(r_filepath, "r")
rem = csv.reader(fr)

term = 12
var = 10

dataframe = []
for line in rem:
    dataframe.append(line)

dim = 1
while len(dataframe) -dim -term >= 1:
    wfntr = path +"\\train" +str(dim) +".csv"
    wfnts = path +"\\test" +str(dim) +".csv"
    fwtr = open(wfntr, "w+", newline="")
    fwte = open(wfnts, "w+", newline="")
    wrmtr = csv.writer(fwtr)
    wrmts = csv.writer(fwte)
    i = 1
    count =0
    while i < len(dataframe) - term - dim + 1:
        k = 0
        temp = []
        while k < dim:
            l = 0
            while l < var:
                temp.append(float(dataframe[i+k][l]))
                l = l + 1
            k = k + 1
        j = 0
        while j < term:
            temp.append(float(dataframe[i+j+dim][1]))
            j = j + 1
        i = i + 1
        if len(dataframe) -dim -i +1 >= term *2:
            wrmtr.writerow(temp)
        elif count >= 11:
            wrmts.writerow(temp)
        else:
            count = count +1
            # wrmts.writerow(temp)
    dim = dim + 1
    fwtr.close()
    fwte.close()
fr.close()

print("program finished")
