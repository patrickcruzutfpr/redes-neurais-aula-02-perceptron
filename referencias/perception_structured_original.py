import numpy as np
import random

def perceptron_simples(X, D, eta=0.1, max_epocas=1000):
    """
    Implementação do Perceptron Simples baseado no pseudocódigo dos slides.
    
    Parâmetros:
        X: matriz de entrada (n_amostras x n_features). Cada linha é uma amostra.
        D: vetor de saídas desejadas (n_amostras,). Valores devem ser 0 ou 1.
        eta: taxa de aprendizado (default=0.1)
        max_epocas: número máximo de épocas (default=1000)
        
    Retorna:
        W: vetor de pesos ajustados (incluindo o bias)
        epocas: número de épocas executadas
        convergiu: True se convergiu, False se atingiu o limite de épocas
    """
    
    # Número de amostras e de features
    n_amostras = X.shape[0]
    n_features = X.shape[1]
    
    # 1. Inicialização: Iniciar o vetor W com valores aleatórios pequenos.
    # Vamos usar a sugestão [-0.5, 0.5]
    W = np.random.uniform(-0.5, 0.5, size=n_features + 1)  # +1 para o bias
    
    # 2. Iniciar o contador de número de épocas
    epocas = 0
    
    # 3. Iniciar variável de controle
    erro = True
    
    # 4. Repetir enquanto (erro == TRUE & epoca < n.Iter)
    while erro and epocas < max_epocas:
        # 5. erro ← FALSE (assumimos que não haverá erro nesta época)
        erro = False
        
        # 6. Para todas as amostras de treinamento em X, fazer:
        for i in range(n_amostras):
            # Criar o vetor de entrada com o bias (x0 = 1)
            x_com_bias = np.insert(X[i], 0, 1)  # Adiciona 1 no início
            
            # 7. V = W' * X // Calcular o sinal do neurônio (spike)
            V = np.dot(W, x_com_bias)
            
            # 8. Y = phi(V) // Calcular o sinal de saída do neurônio (Y)
            # Função de ativação: função degrau (signo)
            if V > 0:
                Y = 1
            else:
                Y = 0
            
            # 9. Se Y (saída obtida) != Di (saída real): // erro na predição
            if Y != D[i]:
                # 10. W = W + η * (Di - Y) * X
                # Note: usamos o vetor x_com_bias aqui, pois o bias também precisa ser atualizado
                W = W + eta * (D[i] - Y) * x_com_bias
                
                # 11. erro ← TRUE (houve pelo menos um erro nesta época)
                erro = True
        
        # 13. epocas ← epocas + 1
        epocas += 1
    
    # Verificar se convergiu
    convergiu = not erro
    
    return W, epocas, convergiu


def prever_perceptron(W, X):
    """
    Função para prever a saída de um Perceptron já treinado.
    
    Parâmetros:
        W: vetor de pesos (incluindo o bias)
        X: matriz de entrada (n_amostras x n_features)
        
    Retorna:
        Y_pred: vetor de saídas previstas
    """
    n_amostras = X.shape[0]
    Y_pred = np.zeros(n_amostras)
    
    for i in range(n_amostras):
        x_com_bias = np.insert(X[i], 0, 1)
        V = np.dot(W, x_com_bias)
        if V > 0:
            Y_pred[i] = 1
        else:
            Y_pred[i] = 0
    
    return Y_pred


# Função principal para testar os datasets
def main():
    print("=== AT 02 - Perceptron Simples ===\n")
    
    # Dataset da porta lógica OR
    print("a) Testando com a porta lógica OR:")
    X_OR = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])
    D_OR = np.array([0, 1, 1, 1])
    
    W_OR, epocas_OR, convergiu_OR = perceptron_simples(X_OR, D_OR, eta=0.1, max_epocas=100)
    print(f"  Pesos finais: {W_OR}")
    print(f"  Épocas necessárias: {epocas_OR}")
    print(f"  Convergiu? {convergiu_OR}")
    Y_pred_OR = prever_perceptron(W_OR, X_OR)
    print(f"  Saídas reais (D): {D_OR}")
    print(f"  Saídas previstas: {Y_pred_OR}")
    print()
    
    # Dataset da porta lógica AND
    print("b) Testando com a porta lógica AND:")
    X_AND = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
    D_AND = np.array([0, 0, 0, 1])
    
    W_AND, epocas_AND, convergiu_AND = perceptron_simples(X_AND, D_AND, eta=0.1, max_epocas=100)
    print(f"  Pesos finais: {W_AND}")
    print(f"  Épocas necessárias: {epocas_AND}")
    print(f"  Convergiu? {convergiu_AND}")
    Y_pred_AND = prever_perceptron(W_AND, X_AND)
    print(f"  Saídas reais (D): {D_AND}")
    print(f"  Saídas previstas: {Y_pred_AND}")
    print()
    
    # Dataset da porta lógica XOR
    print("c) Testando com a porta lógica XOR:")
    X_XOR = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    D_XOR = np.array([0, 1, 1, 0])
    
    W_XOR, epocas_XOR, convergiu_XOR = perceptron_simples(X_XOR, D_XOR, eta=0.1, max_epocas=100)
    print(f"  Pesos finais: {W_XOR}")
    print(f"  Épocas necessárias: {epocas_XOR}")
    print(f"  Convergiu? {convergiu_XOR}")
    Y_pred_XOR = prever_perceptron(W_XOR, X_XOR)
    print(f"  Saídas reais (D): {D_XOR}")
    print(f"  Saídas previstas: {Y_pred_XOR}")
    print("\n--- OBSERVAÇÃO ---")
    print("O Perceptron Simples NÃO consegue aprender a função XOR porque ela NÃO é linearmente separável.")
    print("Isso é esperado e demonstra um dos limites fundamentais do modelo.\n")

# Executa o programa
if __name__ == "__main__":
    main()