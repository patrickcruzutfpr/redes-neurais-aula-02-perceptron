import numpy as np

def perceptron_simples(X, D, eta=0.1, max_epocas=1000):
    n_amostras = X.shape[0]
    n_features = X.shape[1]
    W = np.random.uniform(-0.5, 0.5, size=n_features + 1)
    epocas = 0
    erro = True
    while erro and epocas < max_epocas:
        erro = False
        for i in range(n_amostras):
            x = np.insert(X[i], 0, 1)
            V = np.dot(W, x)
            Y = 1 if V > 0 else 0
            if Y != D[i]:
                W = W + eta * (D[i] - Y) * x
                erro = True
        epocas += 1
    return W, epocas, not erro


def prever_perceptron(W, X):
    n_amostras = X.shape[0]
    Y = np.zeros(n_amostras, dtype=int)
    for i in range(n_amostras):
        x = np.insert(X[i], 0, 1)
        V = np.dot(W, x)
        Y[i] = 1 if V > 0 else 0
    return Y


def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    OR = np.array([0,1,1,1])
    AND = np.array([0,0,0,1])
    XOR = np.array([0,1,1,0])

    W, e, c = perceptron_simples(X, OR, eta=0.1, max_epocas=100)
    print('OR', W, e, c, prever_perceptron(W, X))

    W, e, c = perceptron_simples(X, AND, eta=0.1, max_epocas=100)
    print('AND', W, e, c, prever_perceptron(W, X))

    W, e, c = perceptron_simples(X, XOR, eta=0.1, max_epocas=100)
    print('XOR', W, e, c, prever_perceptron(W, X))


if __name__ == '__main__':
    main()