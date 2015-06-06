import sys

def mle_estimate(m, N, x, y):

    # prob_y[0] = P(Y = 0)
    # prob_y[1] = P(Y = 1)

    # prob_x_y[0][0] = P(Xi = 0, Y = 0)/P(Y = 0)
    # prob_x_y[1][0] = P(Xi = 1, Y = 0)/P(Y = 0)

    #instance_x_y[n][m] = # of instances where Xi = n and Y = m

    #inst = instances in class 0

    prob_y = []
    prob_x_y = dict()
    #instance_x_y = []
    zero_instances = 0
    total_instances = N

    for cl in y:
        # cl = class (0 or 1)
        if cl == 0:
            zero_instances+=1

    prob_y.append(float(zero_instances)/total_instances)
    prob_y.append(float(total_instances - zero_instances)/total_instances) # = 1 - prob_y[0]

    for col in xrange(m):

        instances_zero_zero = 0.0
        instances_zero_one = 0.0
        instances_one_zero = 0.0
        instances_one_one = 0.0

        for row in xrange(N):

            instance = x[row][col]
            classY =  y[row]
            #instances_zero_zero = instances where Xi = 0 and class Y = 0

            if(instance == 0 && classY == 0):
                instances_zero_zero += 1
            if(instance == 0 && classY == 1):
                instances_zero_one += 1
            if(instance == 1 && classY == 0):
                instances_one_zero += 1
            if(instance == 1 && classY == 1):
                instances_one_one += 1
        # N = number of total instances
        # P(Xi = 0, Y = 1) = instances_zero_one/N
        # P(Xi = 0| Y = 1) = P(Xi = 0, Y = 1)/P(Y = 1)
        prob_x_y[col] = [[(instances_zero_zero/N)/prob_y[0],\
                            (instances_zero_one/N)/prob_y[1]],\
                          [(instances_one_zero/N)/prob_y[0], \
                              (instances_one_one/N)/prob_y[1]]]


if __name__ == '__main__' :
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    #lect 24 p. 11

    # m = number of input variables (num of columns)
    # N = number of data vectors (num of rows)
    # x = input variables/observations (matrix)
        #eg.: [[0,0],[0,1],[1,0]]
    # y = output

    m, N, x, y = load_data(train_file)

    # lect 24 p. 24
    #Training:

    # Y = argmax P(X,Y) = argmax P(X|Y)P(Y)

    #mle_x_probs = P(X|Y)
    #mle_y_probs = P(Y)

    mle_x_probs, mle_y_probs = mle_estimate(m, N, x, y);
