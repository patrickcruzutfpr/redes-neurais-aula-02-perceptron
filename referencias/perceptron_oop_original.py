import numpy as np

# A classe PerceptronSimples implementa o modelo de um único neurônio
# com a Regra Delta de aprendizado e a função de ativação degrau.
class PerceptronSimples:
    """
    Implementação do Perceptron Simples usando a Regra de Aprendizado do Perceptron (Regra Delta).
    Projetado para problemas de classificação binária linearmente separáveis.
    """

    def __init__(self, taxa_aprendizado=0.1, n_epocas=100, semente=None):
        """
        Inicializa o Perceptron.

        Args:
            taxa_aprendizado (float): Taxa (η) usada para ajustar os pesos.
            n_epocas (int): Número de vezes que o algoritmo passará por todo o dataset.
            semente (int, optional): Semente para a geração de pesos aleatórios.
        """
        self.taxa_aprendizado = taxa_aprendizado
        self.n_epocas = n_epocas
        self.semente = semente
        self.pesos = None
        self.erros_por_epoca = []

    def _funcao_ativacao(self, net):
        """
        Função de ativação Degrau Bipolar/Heaviside.
        Retorna 1 se a entrada líquida (net) for não-negativa, e 0 caso contrário.

        Args:
            net (float): Entrada líquida (soma ponderada).

        Returns:
            int: A saída binária (0 ou 1).
        """
        return np.where(net >= 0.0, 1, 0)

    def fit(self, X, y):
        """
        Treina o Perceptron Simples usando o algoritmo da Regra Delta.

        Args:
            X (np.ndarray): Matriz de dados de entrada (features).
            y (np.ndarray): Vetor de rótulos alvo (target).
        """
        if self.semente is not None:
            np.random.seed(self.semente)

        # Adiciona uma coluna de '1' ao vetor X para representar a entrada de BIAS (x0)
        # O peso correspondente a essa coluna será o bias (w0)
        X_com_bias = np.insert(X, 0, 1, axis=1)
        n_features = X_com_bias.shape[1]

        # Inicializa os pesos (incluindo o bias/w0) com valores pequenos e aleatórios
        self.pesos = np.random.rand(n_features) * 0.01

        print(f"Pesos iniciais ({n_features} pesos, incluindo BIAS): {self.pesos}")
        print("-" * 50)

        # Loop de treinamento (Iteração por ÉPOCA)
        for epoca in range(1, self.n_epocas + 1):
            erros = 0
            
            # Loop sobre cada amostra de treinamento
            for x_amostra, alvo in zip(X_com_bias, y):
                # 1. Calcular a entrada líquida (net)
                # net = soma(peso * entrada)
                net = np.dot(x_amostra, self.pesos)

                # 2. Obter a saída prevista
                y_previsto = self._funcao_ativacao(net)

                # 3. Calcular o erro
                erro = alvo - y_previsto

                # 4. Se houver erro, ajustar os pesos (Regra Delta)
                if erro != 0:
                    # w_novo = w_antigo + (taxa_aprendizado * erro * entrada)
                    ajuste = self.taxa_aprendizado * erro * x_amostra
                    self.pesos += ajuste
                    erros += 1
            
            self.erros_por_epoca.append(erros)

            # Condição de parada: se não houver erros em uma época, o modelo convergiu
            if erros == 0:
                print(f"Convergiu na Época {epoca}!")
                break
            
            if epoca % 10 == 0 or erros > 0:
                 print(f"Época {epoca}: {erros} erros detectados. Pesos atuais: {self.pesos.round(4)}")

    def predict(self, X):
        """
        Faz previsões em novos dados.

        Args:
            X (np.ndarray): Matriz de dados de entrada (features).

        Returns:
            np.ndarray: Vetor de previsões (0 ou 1).
        """
        # Adiciona o bias (entrada x0=1) para consistência com o treinamento
        X_com_bias = np.insert(X, 0, 1, axis=1)
        
        # net = soma(peso * entrada) para todas as amostras
        net = np.dot(X_com_bias, self.pesos)
        
        # Aplica a função de ativação
        return self._funcao_ativacao(net)

# --- Funções de Ajuda para Execução e Teste ---

def executar_teste(nome_porta, X, y, n_epocas=100, semente=42):
    """
    Função auxiliar para testar o Perceptron em diferentes datasets.
    """
    print("=" * 70)
    print(f"TESTE PARA A PORTA LÓGICA: {nome_porta}")
    print("=" * 70)
    
    # Exibe os dados de entrada e alvo
    print("\n[Dataset de Treinamento]")
    print(f"{'X1':>5} | {'X2':>5} | {'Target (D)':>10}")
    print("-" * 23)
    for x_i, y_i in zip(X, y):
        print(f"{x_i[0]:>5} | {x_i[1]:>5} | {y_i:>10}")
        
    # Inicializa e treina o modelo
    modelo = PerceptronSimples(taxa_aprendizado=0.1, n_epocas=n_epocas, semente=semente)
    modelo.fit(X, y)

    # Faz a previsão com os mesmos dados de treinamento para verificar a acurácia
    previsoes = modelo.predict(X)

    print("\n[Resultado Final do Treinamento]")
    # Exibe os pesos finais, onde w[0] é o BIAS (w0)
    print(f"Pesos Finais (BIAS, w1, w2): {modelo.pesos.round(4)}")
    
    # Exibe a tabela de resultados
    print(f"\n{'X1':>5} | {'X2':>5} | {'Alvo':>5} | {'Previsto':>8} | {'Acerto':>6}")
    print("-" * 38)
    acertos = 0
    for x_i, alvo, prev in zip(X, y, previsoes):
        acertou = 'SIM' if alvo == prev else 'NÃO'
        if acertou == 'SIM':
            acertos += 1
        print(f"{x_i[0]:>5} | {x_i[1]:>5} | {alvo:>5} | {prev:>8} | {acertou:>6}")

    acuracia = (acertos / len(y)) * 100
    print("-" * 38)
    print(f"Acurácia: {acuracia:.2f}%")
    
    if acuracia == 100.0:
        print("Resultado: O Perceptron CONVERGIU (Problema linearmente separável).")
    else:
        print("Resultado: O Perceptron NÃO CONVERGIU (Problema não linearmente separável, como esperado para XOR).")
    
    # Opcional: Mostrar a curva de erro (para análise)
    print(f"\nErros por Época: {modelo.erros_por_epoca}")
    print("\n" * 2)

# --- 1. Definição dos Datasets (Portas Lógicas) ---

# Dataset base: Entradas X1 e X2
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# a) Porta Lógica OR (Linearmente Separável)
y_or = np.array([0, 1, 1, 1])

# b) Porta Lógica AND (Linearmente Separável)
y_and = np.array([0, 0, 0, 1])

# c) Porta Lógica XOR (NÃO Linearmente Separável)
y_xor = np.array([0, 1, 1, 0])

# --- 2. Execução dos Testes ---

# a) Teste para Porta OR
executar_teste("OR", X, y_or)

# b) Teste para Porta AND
executar_teste("AND", X, y_and)

# c) Teste para Porta XOR (A falha é esperada, pois o Perceptron Simples não consegue resolver XOR)
executar_teste("XOR", X, y_xor, n_epocas=200) # Aumentamos um pouco as épocas para garantir que não converge

# Opcional: Teste com semente diferente (dica da atividade)
# executar_teste("OR (Semente Diferente)", X, y_or, semente=100)
