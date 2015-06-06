#! /usr/bin/env python

import sys
from math import exp

def load_data(file_name):

    data = list(open(file_name, 'r'))

    m = int(data[0]) #number of input variables
    N = int(data[1]) #number of data vectors

    x = []
    y = []
    for vec in data[2:]:
        instr, outstr = vec.split(':')
        invec = [1] + [int(_) for _ in instr.split()]
        x.append(invec)  #insert a 1 at index 0 of the row
        y.append(int(outstr))

    return m, N, x, y


def calculate_z(beta, x, m, i):
    z = sum([beta[j]*x[i - 1][j] for j in xrange(m + 1)])
    #for j in xrange(m + 1):
    #    z += beta[j]*x[i-1][j]
    return z

#Training for Logistic Regression

def train_logistic_reg(m, N, x, y, learning_rate):
    beta = [0]*(m + 1) #list of size m that is initialized with all zeros
    epochs = 10000 #number of passes over data during learning (constant)
    #learning_rate = learning rate "mi"

    for e in xrange(epochs):

        gradient = [0]*(m + 1)

        z = []

        for i in xrange(1,N + 1): # i goes from 1 to N (both included)
            z.append(calculate_z(beta, x, m, i))

        for k in xrange(m + 1):
            for i in xrange(1,N + 1): # i goes from 1 to N (both included)
                #  z = calculate_z(beta, x, m, i)
                #for j in xrange(m + 1):
                #    z += beta[j]*x[i-1][j]

               # exp(-z) = e^(-z)
                #print "z is %f ; gradient is %f" %(z, gradient[k])
                gradient[k] += x[i - 1][k]*(y[i - 1] - 1/(1+ exp(-z[i - 1])))

        for k in xrange(m + 1):
            beta[k] += learning_rate*gradient[k]

    return beta

# lect 24, p. 35

def classifier_logistic_reg(beta, x, m, N):
    pred_y = []
    for x_row in xrange(1, N + 1): #goes from 1 to N (both included)
        z = calculate_z(beta, x, m, x_row)
        # p = P(Y = 1|X)
        p = 1/(1+exp(-z))

        if p > 0.5:
            pred_y.append(1)
        else:
            pred_y.append(0)
    return pred_y

# pred_y = values of y we predicted
# y = actual values of y

def calculate_accuracy(pred_y, y):
    total_zero = 0
    total_one = 0
    correct_zero = 0
    correct_one = 0

    for i in xrange(len(y)):

        if y[i] == 0:
            total_zero+=1

        if y[i] == 1:
            total_one+=1

        if y[i] == pred_y[i]:
            if y[i] == 0:
                correct_zero += 1
            if y[i] == 1:
                correct_one += 1
    accuracy = float(correct_zero + correct_one)/(total_zero + total_one)

    print "Class 0: tested %d, correctly classified %d" %(total_zero, correct_zero)
    print "Class 1: tested %d, correctly classified %d" %(total_one, correct_one)
    print "Accuracy = %1.2f" %(accuracy)


if __name__ == '__main__' :
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    m, N, x, y = load_data(train_file)

    beta = train_logistic_reg(m, N, x, y, 0.0000101)

    print beta

    m, N, x, y = load_data(test_file)

    pred_y = classifier_logistic_reg(beta, x, m, N)

    print pred_y

    calculate_accuracy(pred_y, y)
