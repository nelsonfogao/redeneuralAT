import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Defina o número de nós nas camadas de entrada, ocultas e de saída.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Inicializa pesos
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        #### TODO: Defina self.activation_function para sua função sigmoide implementada ####
        #
        # Observação: Em Python, você pode definir uma função com uma expressão lambda
        #self.activation_function = 0  # Substitua 0 pelo seu cálculo sigmoide.

        '''
        Se o código lambda não for algo com o qual você esteja familiarizado, 
        você pode remover o comentário das três linhas a seguir e colocar sua implementação lá.
        '''
        def sigmoid(x):
           return 1/(1+ np.exp(-x))  # Replace 0 with your sigmoid calculation here
        self.activation_function = sigmoid

    def train(self, features, targets):
        ''' Treine a rede em um lote de características e rótulos.

            Argumentos
            ---------

            features: Array 2D, cada linha é uma amostra de dados, cada coluna é uma característica
            targets: Matriz 1D de valores de rótulo

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implementar o forward pass na função abaixo
            # Implementar o backpropagation
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Implemente o forward pass aqui

            Argumentos
            ---------
            X: lote de características

        '''
        #### Sua implementação aqui ####
        ### Forward pass ###
        # TODO: Hidden layer - Substitua esses valores pelos seus cálculos.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # sinais sinápticos para a camada escondida
        hidden_outputs = self.activation_function(hidden_inputs)

        # TODO: Output layer - Substitua esses valores pelos seus cálculos.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # sinais para a camada de saída
        final_outputs = final_inputs  # Aqui você poe uma função de ativação para o caso de classificação

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implemente o backpropagation

            Argumentos
            ---------
            final_outputs: saída do forward pass
            y: lote de rótulos
            delta_weights_i_h: alteração nos pesos da entrada para as camadas ocultas
            delta_weights_h_o: alteração nos pesos das camadas ocultas para as de saída

        '''
        #### Implemente o backpropagation aqui ####
        ### Backpropagation ###

        # TODO: Output error - Substitua esses valores pelos seus cálculos.
        error = y - final_outputs  # O erro da camada de saída é a diferença entre o destino desejado e a saída real.
        output_error_term = error.reshape((1, error.shape[0]))

        # Obs.: A derivada da função de ativação sigmoide é  (1 - sigmoide) -> (1 - hidden_outputs)
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, error)
        hidden_error_term = hidden_error * (hidden_outputs * (1 - hidden_outputs))

        # Atualizar os deltas (input to hidden)
        # TODO: Atualize os deltas dos pesos
        delta_weights_i_h += hidden_error_term.T * X[:,None]
        # Atualizar os deltas (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Atualize os pesos na etapa de gradiente descendente

            Arguments
            ---------
            delta_weights_i_h: alteração nos pesos da entrada para as camadas ocultas
            delta_weights_h_o: alteração nos pesos das camadas ocultas para as de saída
            n_records: número de registros

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # atualize os pesos ocultos para a saída com a etapa de gradiente descendente
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records  # atualize os pesos de entrada para ocultos com passo de gradiente descendente

    def run(self, features):
        ''' Executa forward pass pela rede com características de entrada

            Argumentos
            ---------
            features: Matriz 1D de valores de recursos
        '''

        #### Implemente o backpropagation aqui ####
        # TODO: Hidden layer - substitua esses valores pelos cálculos apropriados.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # sinais na camada oculta
        hidden_outputs = self.activation_function(hidden_inputs)  # sinais da camada oculta

        # TODO: Output layer - substitua esses valores pelos cálculos apropriados.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # sinais na camada de saída final
        final_outputs = final_inputs  # De novo, aqui você poe uma função de ativação para o caso de classificação

        return final_outputs

