#! /usr/bin/env python

import numpy as np
import sys

def mle_estimate(m, N, x, y):
    total = len(y)
    
    #Determine the P(Y)
    one_counts = np.count_nonzero(y)
    zero_counts = total - one_counts 
    class_probs = np.array([zero_counts, one_counts], dtype=float)/total
    
    #Determine P(Xi, Y)
    x_probs = np.zeros((4, m))
    for i in xrange(m):
        # x_i ==0  and y==0
        counts_00 = np.sum(np.logical_and(x[:,i] == 0, y == 0), dtype=float)
        # x_i ==0  and y==1
        counts_01 = np.sum(np.logical_and(x[:,i] == 0, y == 1), dtype=float)
        # x_i ==1  and y==0
        counts_10 = np.sum(np.logical_and(x[:,i] == 1, y == 0), dtype=float)
        # x_i ==1  and y==1
        counts_11 = np.sum(np.logical_and(x[:,i] == 1, y == 1), dtype=float)
        x_probs[:,i] = np.array([counts_00, counts_01, counts_10, counts_11])/total
                      
    return np.array(x_probs), class_probs

def map_estimate(m, N, x, y):
    total = len(y)
    
    #Determine the P(Y)
    one_counts = np.count_nonzero(y) + 2
    zero_counts = total - one_counts + 2
    class_probs = np.array([zero_counts, one_counts], dtype=float)/(total + 4)
    
    #Determine P(Xi, Y)
    x_probs = np.zeros((4, m))
    for i in xrange(m):
        # x_i ==0  and y==0
        counts_00 = np.sum(np.logical_and(x[:,i] == 0, y == 0), dtype=float) + 1
        # x_i ==0  and y==1
        counts_01 = np.sum(np.logical_and(x[:,i] == 0, y == 1), dtype=float) + 1
        # x_i ==1  and y==0
        counts_10 = np.sum(np.logical_and(x[:,i] == 1, y == 0), dtype=float) + 1
        # x_i ==1  and y==1
        counts_11 = np.sum(np.logical_and(x[:,i] == 1, y == 1), dtype=float) + 1
        x_probs[:,i] = np.array([counts_00, counts_01, counts_10, counts_11])/(total+ 4)
                      
    return np.array(x_probs), class_probs

def classify(m, N, x, x_probs, y_probs):
    y_hat = np.zeros(N)
    #Classify each data vector in the x set
    for i in xrange(N):
        #Test y = 0
        class_1 = y_probs[0]
        class_2 = y_probs[1]
        for j in xrange(m):
            probabilities = np.reshape(x_probs[:,j].view(),(2,2))
            class_1 *= probabilities[x[i,j],0]
            class_2 *= probabilities[x[i,j],1]
        y_hat[i] = 0 if class_1 > class_2 else 1
    return y_hat

def load_data(file_name):
    
    data = list(open(file_name, 'r'))
    
    m = int(data[0]) #number of input variables
    N = int(data[1]) #number of data vectors
    
    x = []
    y = []
    for vec in data[2:]:
        instr, outstr = vec.split(':')
        invec = [int(_) for _ in instr.split()]
        x.append(invec)
        y.append(int(outstr))
        
    return m, N, np.array(x), np.array(y) 
 
def give_stats(y, y_hat):
    classes = [0,1]
    total = len(y)
    yone_counts = np.count_nonzero(y)
    yzero_counts = total - yone_counts 
    accurate_zero= np.sum(np.logical_and(y==0, y_hat==0))
    print "Class 0: tested {0}, correctly classified {1}".format(yzero_counts, accurate_zero)
    accurate_one = np.sum(np.logical_and(y==1, y_hat==1))
    print "Class 1: tested {0}, correctly classified {1}".format(yone_counts, accurate_one)
    total_accurate = accurate_zero + accurate_one
    print "Overall: tested {0}, correctly classified {1}".format(total, total_accurate)
    accuracy = float(total_accurate)/total
    print "Accuracy = {0}".format(accuracy)


if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    m, N, x, y = load_data(train_file)
    
    mle_x_probs, mle_y_probs = mle_estimate(m, N, x, y)
    map_x_probs, map_y_probs = map_estimate(m, N, x, y)
    
    m, N, x, y = load_data(test_file)
    y_hat = classify(m, N, x, mle_x_probs, mle_y_probs)
    print "\nUsing MLE:\n"
    give_stats(y, y_hat)

    y_hat = classify(m, N, x, map_x_probs, map_y_probs)
    print "\nUsing MAP:\n"
    give_stats(y, y_hat)

    