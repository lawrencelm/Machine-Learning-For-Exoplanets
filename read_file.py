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

print 
train_file = input("Enter the training file name:")
m, N, x, y = load_data(train_file)
print m, N
print x
print y

test_file = input("Enter the test file name:")
m, N, x, y = load_data(train_file)
print m, N
print x
print y