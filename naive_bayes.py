import sys

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

    return m, N, x, y

#Training through Laplace Estimators or MAP
# lect 24, p. 52

def laplace_estimate(m, N, x, y):

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

    one_instances = total_instances - zero_instances
    prob_y.append(float(zero_instances + 1)/(total_instances + 2))
    prob_y.append(float(one_instances + 1)/(total_instances + 2)) # = 1 - prob_y[0]

    for col in xrange(m):



        instances_zero_zero = 0.0
        instances_zero_one = 0.0
        instances_one_zero = 0.0
        instances_one_one = 0.0

        for row in xrange(N):

            instance = x[row][col]
            classY = y[row]
            #instances_zero_zero = instances where Xi = 0 and class Y = 0

            if(instance == 0 and classY == 0):
                instances_zero_zero += 1
            if(instance == 0 and classY == 1):
                instances_zero_one += 1
            if(instance == 1 and classY == 0):
                instances_one_zero += 1
            if(instance == 1 and classY == 1):
                instances_one_one += 1

        # N = number of total instances
        # P(Xi = 0, Y = 1) = instances_zero_one/N
        # prob_x_y[col][0][1] = P(Xi = 0| Y = 1) = P(Xi = 0, Y = 1)/P(Y = 1)
        # col = number of the column

        prob_x_y[col] = [[(instances_zero_zero + 1)/(zero_instances + 2),\
                            (instances_zero_one + 1)/(one_instances + 2)],\
                          [(instances_one_zero + 1)/(zero_instances + 2), \
                              (instances_one_one + 1)/(one_instances + 2)]]
        print (instances_one_one + 1)/(one_instances + 2)
    return prob_x_y, prob_y



#Training through Maximum Likelihood Estimators

def mle_estimate(m, N, x, y):

    # prob_y[0] = P(Y = 0)
    # prob_y[1] = P(Y = 1)

    # prob_x_y[col][0][0] = P(Xi = 0, Y = 0)/P(Y = 0)
    # prob_x_y[col][1][0] = P(Xi = 1, Y = 0)/P(Y = 0)

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

    prob_y.append(float(zero_instances)/(total_instances))
    prob_y.append(float(total_instances - zero_instances)/total_instances) # = 1 - prob_y[0]

    for col in xrange(m):

        instances_zero_zero = 0.0
        instances_zero_one = 0.0
        instances_one_zero = 0.0
        instances_one_one = 0.0

        for row in xrange(N):

            instance = x[row][col]
            classY = y[row]
            #instances_zero_zero = instances where Xi = 0 and class Y = 0

            if(instance == 0):# and classY == 0):
                instances_zero_zero += 1
            if(instance == 0):# and classY == 1):
                instances_zero_one += 1
            if(instance == 1):# and classY == 0):
                instances_one_zero += 1
            if(instance == 1):# and classY == 1):
                instances_one_one += 1

        # N = number of total instances
        # P(Xi = 0, Y = 1) = instances_zero_one/N
        # prob_x_y[col][0][1] = P(Xi = 0| Y = 1) = P(Xi = 0, Y = 1)/P(Y = 1)
        # col = number of the column

        prob_x_y[col] = [[(instances_zero_zero/N)/prob_y[0],\
                            (instances_zero_one/N)/prob_y[1]],\
                          [(instances_one_zero/N)/prob_y[0], \
                              (instances_one_one/N)/prob_y[1]]]

    return prob_x_y, prob_y

#Classifying
#lect 14, p. 26

def bayes_predictor(x, m, N, prob_x_y, prob_y):

    pred_y = []

    for x_row in xrange(N):

        # pred_y_one = Y hat for Y = 1

        pred_y_one = 1
        pred_y_zero = 1

        for x_col in xrange(m):
            x_value = x[x_row][x_col]
            pred_y_zero *= prob_x_y[x_col][x_value][0]
            pred_y_one *= prob_x_y[x_col][x_value][1]

        pred_y_zero *= prob_y[0]
        pred_y_one *= prob_y[1]

        #y_hat is the value of y that has higher likelihood
        y_hat = 0 if pred_y_zero > pred_y_one else 1

        pred_y.append(y_hat)

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

    print "\nUsing Maximum Likelihood Estimator\n"

    prob_x_y, prob_y = mle_estimate(m, N, x, y);

  #  print prob_x_y
  #  print prob_y

    m, N, x, y = load_data(test_file)

    #pred_y is an array of predicted values of Y for each vector
    pred_y = bayes_predictor(x, m, N, prob_x_y, prob_y)

  #  print pred_y

    calculate_accuracy(pred_y, y);

    #Now using Laplace Estimators instead of MLE

    #m, N, x, y = load_data(train_file)

    print "\nUsing Laplace Estimate\n"

    prob_x_y, prob_y = laplace_estimate(m, N, x, y);

  #  print prob_x_y
  #  print prob_y

    m, N, x, y = load_data(test_file)

    #pred_y is an array of predicted values of Y for each vector
    pred_y = bayes_predictor(x, m, N, prob_x_y, prob_y)

  #  print pred_y

    calculate_accuracy(pred_y, y);

