import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

def Separar(dataset):
    X = dataset[:,:-1]
    Y = dataset[:,-1]
    return X, Y

def sigmoid(X):
    return 1/(1 + np.exp(-X))

def invsigmoid(X):
    return np.log(X / (1 - X))

class Metric(object):
    def __call__(self, target, prediction):
        return NotImplemented


class MSE(Metric):
    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n

def Red3MiniBatch(X_train, Y_train, alpha=0.01, epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 16
    n,m = X_train.shape

    # Inicializo los pesos para las 3 capas
    W1 = np.random.randn(m * 3).reshape(m, 3)
    b1 = np.random.randn(3).reshape(3, 1)
    W2 = np.random.randn(3 * 2).reshape(3, 2)
    b2 = np.random.randn(2).reshape(2, 1)
    W3 = np.random.randn(2 * 1).reshape(2, 1)
    b3 = np.random.randn(1).reshape(1, 1)

    for i in range(epochs):
        idx = np.random.permutation(n)
        X_train = X_train[idx]
        Y_train = Y_train[idx]
        batch_size = int(len(X_train) / b)

        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            Batch_X = X_train[i: end]
            Batch_Y = Y_train[i: end]

            #Forward Step
            Z1 = np.matmul(Batch_X, W1) + b1.T
            A1 = sigmoid(Z1)
            Z2 = np.matmul(A1, W2) + b2.T
            A2 = sigmoid(Z2)
            Z3 = np.matmul(A2, W3) + b3.T
            Y_hat = sigmoid(Z3)
            err = Batch_Y - Y_hat

            #Backward Step
            aux3 = np.multiply(sigmoid(Z3), 1 - sigmoid(Z3))
            dZ3 = -2*np.matmul(err,aux3)
            gradW3 = np.matmul(A2.T, dZ3) / batch_size
            gradb3 = np.sum(dZ3, axis=0, keepdims=True)

            aux2 = np.multiply(sigmoid(Z2), 1 - sigmoid(Z2))
            dZ2 = np.multiply(np.matmul(dZ3, W3.T),aux2)
            gradW2 = np.matmul(A1.T, dZ2) / batch_size
            gradb2 = np.sum(dZ2, axis=0, keepdims=True)

            aux1 = np.multiply(sigmoid(Z1), 1 - sigmoid(Z1))
            dZ1 = np.multiply(np.matmul(dZ2, W2.T), aux1)
            gradW1 = np.matmul(Batch_X.T, dZ1) / batch_size
            gradb1 = np.sum(dZ1, axis=0, keepdims=True)

            #Update
            W3 = W3 - alpha * gradW3
            b3 = b3 - alpha * gradb3.T
            W2 = W2 - alpha * gradW2
            b2 = b2 - alpha * gradb2.T
            W1 = W1 - alpha * gradW1
            b1 = b1 - alpha * gradb1.T

    return W1,b1,W2,b2,W3,b3

train = genfromtxt('clase_2_train_data.csv', delimiter=',')
test = genfromtxt('clase_2_test_data.csv', delimiter=',')
X_train,Y_train = Separar(train)
X_test,Y_test = Separar(test)

X1 = X_train[Y_train==1]
X0 = X_train[Y_train==0]
# X1 = X_test[Y_test==1]
# X0 = X_test[Y_test==0]
plt.scatter(X0[:,0], X0[:,1], color='r')
plt.scatter(X1[:,0], X1[:,1], color='g')
plt.show()

mse = MSE()
W1,B1,W2,B2,W3,B3 = Red3MiniBatch(X_train,Y_train,alpha=0.0001,epochs=2000)

Z1 = np.matmul(X_test,W1) + B1.T
A1 = sigmoid(Z1)
Z2 = np.matmul(A1,W2) + B2.T
A2 = sigmoid(Z2)
Yest = np.matmul(A2,W3) + B3.T
ECM = mse(Yest, Y_test)
print(ECM)

