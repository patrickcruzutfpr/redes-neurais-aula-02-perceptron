import numpy as np
import matplotlib.pyplot as plt

def plot_dados(X, D, titulo):
    plt.figure(figsize=(5,5))
    for i in range(len(X)):
        if D[i] == 0:
            plt.plot(X[i,0], X[i,1], 'ro')
        else:
            plt.plot(X[i,0], X[i,1], 'bo')
    plt.grid(True)
    plt.title(titulo)
    plt.xlabel('X1')
    plt.ylabel('X2')

def plot_decisao(X, D, W, titulo):
    plt.figure(figsize=(5,5))
    
    # Plot dos pontos
    for i in range(len(X)):
        if D[i] == 0:
            plt.plot(X[i,0], X[i,1], 'ro')
        else:
            plt.plot(X[i,0], X[i,1], 'bo')
    
    # Plot da fronteira de decisão
    x1 = np.linspace(-0.5, 1.5, 100)
    x2 = -(W[0] + W[1]*x1)/W[2]
    plt.plot(x1, x2, 'g-')
    
    plt.grid(True)
    plt.title(titulo)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.axis([-0.5, 1.5, -0.5, 1.5])

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

    # Porta OR
    plot_dados(X, OR, 'Distribuição Inicial - OR')
    W, e, c = perceptron_simples(X, OR, eta=0.1, max_epocas=100)
    print('OR:', 'pesos:', W, 'épocas:', e, 'convergiu:', c)
    plot_decisao(X, OR, W, 'Fronteira de Decisão - OR')

    # Porta AND
    plot_dados(X, AND, 'Distribuição Inicial - AND')
    W, e, c = perceptron_simples(X, AND, eta=0.1, max_epocas=100)
    print('AND:', 'pesos:', W, 'épocas:', e, 'convergiu:', c)
    plot_decisao(X, AND, W, 'Fronteira de Decisão - AND')

    # Porta XOR
    plot_dados(X, XOR, 'Distribuição Inicial - XOR')
    W, e, c = perceptron_simples(X, XOR, eta=0.1, max_epocas=100)
    print('XOR:', 'pesos:', W, 'épocas:', e, 'convergiu:', c)
    plot_decisao(X, XOR, W, 'Fronteira de Decisão - XOR')
    
    plt.show()


if __name__ == '__main__':
    main()