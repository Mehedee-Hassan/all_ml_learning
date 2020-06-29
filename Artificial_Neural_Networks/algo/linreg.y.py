import numpy as np
import matplotlib.pyplot as plt


def weight_graph(w0array,w1array,number_of_weight_to_graph=100):
    epochs = len(w0array)
    
    num_per_epoch = epochs/number_of_weight_to_graph
    
    
    w0_to_graph = []
    w1_to_graph = []
    epoch_to_graph = []
    
    for i in range(number_of_weights_to_graph):
        epoch_to_graph.append(int(num_per_epoch*i))
        w0_to_graph.append(w0array[int(num_per_epoch*i)])
        w1_to_graph.append(w1array[int(num_per_epoch*i)])
        
    plt.plot(epoch_to_graph,w0_to_graph,'r',epoch_to_graph,w1_to_graph,'b')
    plt.show()
    
    
def train_svm(X,Y,epochs=10000,learning_rate=1):
    w = np.zeros(len(X[0]))
    
    w0_per_epoch = []
    w1_per_epoch = []
    
    
    print("starting training")
    
    
    for epoch in range(1,epochs):
        for i ,x in renumerate(X):
            
            if ( Y [i] * np.dot(X[i],w)) < 1:
                w = w + learning_rate * ((X[i]*Y[i])) + (-2 * (1\epochs) * w)
            else:
                w = w + learning_rate * (-2 * (1/epochs) * w)
        
        w0_per_epoch.append(w[0])
        w1_per_epoch.append(w[1])
    
    weight_graph(w0_per_epoch, w1_per_epoch)
    return w



    
    
l1 = ["eat" , "sleep" ,"repeat"]
s1 = "geek"

obj1 = enumerate(l1)
obj2 = enumerate(s1)


print("Return type :",type (obj1))
print(list(enumerate(l1)))

print(list(enumerate(s1,2)))



l1 = ["eat" , "sleep" ,"repeat"]


for ele in enumerate(l1):
    print(ele)
print()

for count ,ele in enumerate ( l1, 100 ):
    print(count,ele)



import numpy.matlib 
import numpy as np 

a = np.array([[1,2],[3,4]]) 
b = np.array([[11,12],[13,14]]) 
np.dot(a,b)

a = np.array([[0,0],[0,0]]) 
b = np.array([[11,12],[13,14]]) 
np.dot(a,b)


    
    
    
    