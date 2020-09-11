import numpy as np


def stochastic_gradient_descent(X, Y, alpha=0.01, epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n,m = X.shape

    # initialize random weights
    #W = np.random.randn(m).reshape(m, 1)
    W1 = np.random.randn(6).reshape(2, 3)
    W2 = np.random.randn(3)

    for i in range(epochs):

        for j in range(n):
            #Forward step
            Z1 = X[j, 0] * W1[0, 0] + X[j, 1] * W1[0, 1] + W1[0, 2]
            Z2 = X[j, 0] * W1[1, 0] + X[j, 1] * W1[1, 1] + W1[1, 2]
            a1 = sigmoid(Z1)
            a2 = sigmoid(Z2)
            Y_hat = a1 * W2[0] + a2 * W2[1] + W2[2]

            #Backward step
            err = Y[j]-Y_hat
            W2[0] = W2[0] + 2*alpha*err*a1
            W2[1] = W2[1] + 2*alpha*err*a2
            W2[2] = W2[2] + 2*alpha*err
            W1[0, 0] = W1[0, 0] + 2 * alpha * err * W2[0] * sigmoid(Z1) * (1 - sigmoid(Z1)) * X[j, 0]
            W1[0, 1] = W1[0, 1] + 2 * alpha * err * W2[0] * sigmoid(Z1) * (1 - sigmoid(Z1)) * X[j, 1]
            W1[0, 2] = W1[0, 2] + 2 * alpha * err * W2[0] * sigmoid(Z1) * (1 - sigmoid(Z1))
            W1[1, 0] = W1[1, 0] + 2 * alpha * err * W2[1] * sigmoid(Z2) * (1 - sigmoid(Z2)) * X[j, 0]
            W1[1, 1] = W1[1, 1] + 2 * alpha * err * W2[1] * sigmoid(Z2) * (1 - sigmoid(Z2)) * X[j, 1]
            W1[1, 2] = W1[1, 2] + 2 * alpha * err * W2[1] * sigmoid(Z2) * (1 - sigmoid(Z2))

    return W1, W2

def sigmoid(X):
    return 1/(1 + np.exp(-X))

class Metric(object):
    def __call__(self, target, prediction):
        return NotImplemented


class MSE(Metric):
    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n

mse = MSE()
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y = np.array([0, 1, 1, 0]).T
W1, W2 = stochastic_gradient_descent(X,Y,alpha=0.05,epochs=2500)

Z = np.matmul(X,W1.T)
a = sigmoid(Z)
ones = np.array([1,1,1,1]).reshape(4,1)
a = np.append(a,ones,axis=1)
Yest = np.matmul(a,W2.T)
lr_mse = mse(Y, Yest)
print(Yest)
print(lr_mse)
