'''
Trabalho 5 - Sistemas Inteligentes
Perceptron - Autenticação de cédulas com 3 variáveis de entrada
Sarah R. L. Carneiro
'''

import random
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn

class Perceptron:

    def __init__(self, learningRate, numberOfTimes, weights):

        self.learningRate = learningRate  # taxa de apredisagem
        self.numberOfTimes = numberOfTimes  # numero de epocas
        self.W = weights  # pesos sinapticos
        self.activationFunc = self.unitStep  # funcao de ativacao

    def testPcn(self, X):

        return self.predict(X)

    def trainPcn(self, X, T, tipeOfTrain):

        # treinamento de batch
        if (tipeOfTrain == 0):

            for i in range(self.numberOfTimes):
                O = self.predict(X)

                # W = W + lambda * X * (T - O)
                self.W += self.learningRate * np.dot(X.T, (T - O))

        # treinamento sequencial
        else:

            for i in range(self.numberOfTimes):
                for k in range(len(X)):
                    O = self.predict(X[k, :])
                    self.W += self.learningRate * np.dot(X[k, :][np.newaxis].T, (T[k, :] - O)[np.newaxis])

    # f(XW - b)
    def predict(self, X):

        h = np.dot(X, self.W)
        return self.activationFunc(h)

    # funcao de ativacao - degrau unitario
    def unitStep(self, x):

        return np.where(x > 0, 1, 0)

    # Adequa os padrões de entrada
    @staticmethod
    def dataTreatment(var1, var2, var3):

        X = np.concatenate((var1, var2, var3), axis=1)

        # normaliza os padrões de entrada para o intervalo [0,1]
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

        # adiciona o bias aos padroes de entrada
        X = np.concatenate((np.full((len(X), 1), -1.0), X), axis=1)

        return X

    @staticmethod
    def calcAccuracy(O, T):

        count = 0
        for i in range(len(T)):
            if (O[i] != T[i]):
                count += 1

        return (1 - count / len(T)) * 100

    @staticmethod
    def loadData():

        file = open('data_banknote_authentication.txt', 'r')

        # variaveis de entrada
        variance = []
        skewness = []
        curtosis = []
        entropy = []

        # alvos
        T = []

        for line in file:

            i = random.randrange(0, 1372, 1)
            line = line.split(',')
            variance.insert(i, float(line[0]))
            skewness.insert(i, float(line[1]))
            curtosis.insert(i, float(line[2]))
            entropy.insert(i, float(line[3]))
            T.insert(i, float(line[4]))

        file.close()

        return np.array([variance]).T, np.array([skewness]).T, np.array([curtosis]).T, np.array([entropy]).T, np.array([T]).T

    @staticmethod
    def plotConfusionMatrix(data, labels, output_filename):
        seaborn.set(color_codes=True)
        plt.figure(1, figsize=(9, 6))

        plt.title("Matriz de confusão")

        seaborn.set(font_scale=1.4)
        ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt="d")

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        #ax.set(ylabel="True Label", xlabel="Predicted Label")

        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

def main():

    print('\n\n===== Perceptron - Autenticação de cédulas com 3 variáveis de entrada =====\n\n')

    learningRate = float(input('Insira a taxa de aprendizagem:'))
    numberOfTimes = int(input('Insira o número de épocas:'))
    percentTrain = float(input('Insira a porcentagem de dados que serão utililizados no treinamento:'))
    tipeOfTrain = int(input('Insira [0] para treinamento de lote e [1] treinamento sequencial:'))

    if (tipeOfTrain != 0 and tipeOfTrain != 1):
        print('Entrada inválida!!')
        exit(0)

    random.seed(2)

    variance, skewness, curtosis, entropy, T = Perceptron.loadData()

    # Padrões de entrada: variancia, assimetria e curtose - subconjunto 1
    X_v_s_c = Perceptron.dataTreatment(variance, skewness, curtosis)

    # Padrões de entrada: assimetria, curtose e entropia - subconjunto 2
    X_s_c_e = Perceptron.dataTreatment(skewness, curtosis, entropy)

    # Padrões de entrada: variancia, assimetria e entropia - subconjunto 3
    X_v_s_e = Perceptron.dataTreatment(variance, skewness, entropy)

    # Padrões de entrada: variancia, curtose e entropia - subconjunto 4
    X_v_c_e = Perceptron.dataTreatment(variance, curtosis, entropy)

    # cria os conjuntos de treinamento e de teste
    idx = int(percentTrain*1372)
    train_v_s_c = X_v_s_c[0:idx, :]
    test_v_s_c = X_v_s_c[idx+1:1372, :]

    train_s_c_e = X_s_c_e[0:idx, :]
    test_s_c_e = X_s_c_e[idx + 1:1372, :]

    train_v_s_e = X_v_s_e[0:idx, :]
    test_v_s_e = X_v_s_e[idx + 1:1372, :]

    train_v_c_e = X_v_c_e[0:idx, :]
    test_v_c_e = X_v_c_e[idx + 1:1372, :]

    # alvos
    trainLabels = T[0:idx, :]
    testeLabels = T[idx + 1:1372, :]

    # pesos sinapticos
    weights = np.random.normal(0, 0.01, (4, 1))

    models = {}

    # Treina e testa o subconjunto 1
    p_v_s_c = Perceptron(learningRate=learningRate, numberOfTimes=numberOfTimes, weights=weights)
    p_v_s_c.trainPcn(train_v_s_c, trainLabels, tipeOfTrain)
    O_v_s_c = p_v_s_c.testPcn(test_v_s_c)
    models[Perceptron.calcAccuracy(O_v_s_c, testeLabels)] = [p_v_s_c, O_v_s_c, test_v_s_c]

    # Treina e testa o subconjunto 2
    p_s_c_e = Perceptron(learningRate=learningRate, numberOfTimes=numberOfTimes, weights=weights)
    p_s_c_e.trainPcn(train_s_c_e, trainLabels, tipeOfTrain)
    O_s_c_e = p_v_s_c.testPcn(test_s_c_e)
    models[Perceptron.calcAccuracy(O_s_c_e, testeLabels)] = [p_s_c_e, O_s_c_e, test_s_c_e]


    # Treina e testa o subconjunto 3
    p_v_s_e = Perceptron(learningRate=learningRate, numberOfTimes=numberOfTimes, weights=weights)
    p_v_s_e.trainPcn(train_v_s_e, trainLabels, tipeOfTrain)
    O_v_s_e = p_v_s_c.testPcn(test_v_s_e)
    models[Perceptron.calcAccuracy(O_v_s_e, testeLabels)] = [p_v_s_e, O_v_s_e, test_v_s_e]


    # Treina e testa o subconjunto 4
    p_v_c_e = Perceptron(learningRate=learningRate, numberOfTimes=numberOfTimes, weights=weights)
    p_v_c_e.trainPcn(train_v_c_e, trainLabels, tipeOfTrain)
    O_v_c_e = p_v_s_c.testPcn(test_v_c_e)
    models[Perceptron.calcAccuracy(O_v_c_e, testeLabels)] = [p_v_c_e, O_v_c_e, test_v_c_e]

    print('Acurárias obtidas:', models.keys())

    # Seleciona o subconjunto que resulta na maior acurácia
    p, O, testSet = models[max(models.keys())]
    print('Maior Acurácia: ', max(models.keys()))

    # plota os padroes de entrada por classe (conjunto de teste)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cores = {0: 'green', 1: 'red'}
    labels = [cores[x] for x in testeLabels[:, 0]]
    ax.scatter(testSet[:, 1], testSet[:, 2], testSet[:, 3], marker='o', c=labels)

    # plota limear de decisão
    X, Y = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))
    Z = (p.W[0, :] - p.W[1,:] * X - p.W[2,:] * Y) / p.W[3,:]
    ax.plot_surface(X, Y, Z, alpha=0.3)
    plt.show()

    conf_mat = confusion_matrix(testeLabels, O)
    print('Matriz de Confusão', conf_mat, len(testeLabels))
    Perceptron.plotConfusionMatrix(conf_mat, ['Nota Falsa', 'Nota Verdadeira'], 'matrizConfusao')


if __name__ == '__main__':
    main()

