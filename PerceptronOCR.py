'''
Trabalho 3 - Sistemas Inteligentes
Perceptron - OCR com 4 neurônios
Sarah R. L. Carneiro
'''

import numpy as np
from tensorflow import keras
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



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

    # Adequa a matriz de alvos para a modelagem com 4 neurônios
    @staticmethod
    def labelTreatment(T):

        binary = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0],
                  [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1]]

        for i in range(0, len(T)):

            if (i == 0):
                auxT = np.array(binary[T[i]], ndmin=2)

            else:
                auxT = np.concatenate((auxT, np.array(binary[T[i]], ndmin=2)), axis=0)

        return auxT

    # Adequa os padrões de entrada
    @staticmethod
    def dataTreatment(X):

        # converte a matriz de 3 dimensões para 2 dimensões
        X = X.reshape(len(X), 784)

        # normaliza os valores de luminância - originalmente valores inteiros
        # no intervalo [0,255] - para valores reais no intervalo [0,1]
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

        # adiciona o bias aos padroes de entrada
        X = np.concatenate((np.full((len(X), 1), -1.0), X), axis=1)

        return X

    # Utilizo o Keras para importar o dataset
    # https://keras.io/api/datasets/mnist/
    @staticmethod
    def loadData():

        dataSet = keras.datasets.mnist
        ((trainSet, trainLabels), (testSet, testLabels)) = dataSet.load_data(path="mnist.npz")

        return trainSet, trainLabels, testSet, testLabels

    @staticmethod
    def binaryToDecimal (M):

        return np.array(list(map(lambda m: 9 if m[0] * 8 + m[1] * 4 + m[2] * 2 + m[3] > 9 else m[0] * 8 + m[1] * 4 + m[2] * 2 + m[3], M)))

    @staticmethod
    def calcAccuracy(O, T):
        count = 0
        for i in range(len(T)):
            if(O[i] != T[i]):
                count+=1

        return (1 - count / len(T)) * 100

def main():

    print('\n\n===== Perceptron - OCR 4 Neurônios =====\n\n')
    learningRate = float(input('Insira a taxa de aprendizagem:'))
    numberOfTimes = int(input('Insira o número de épocas:'))
    tipeOfTrain = int(input('Insira [0] para treinamento de lote e [1] treinamento sequencial:'))

    if (tipeOfTrain != 0 and tipeOfTrain != 1):
        print('Entrada inválida!!')
        exit(0)

    # carrega os padroes de entrada e alvos
    trainSet, trainLabels, testSet, testLabels = Perceptron.loadData()

    # padroes de entrada
    trainSet = Perceptron.dataTreatment(trainSet)
    testSet = Perceptron.dataTreatment(testSet)

    # alvos
    trainLabels = Perceptron.labelTreatment(trainLabels)

    # pesos sinapticos
    weights = np.random.normal(0, 0.1, (785, 4))

    p = Perceptron(learningRate=learningRate, numberOfTimes=numberOfTimes, weights=weights)

    # treina o perceptron
    p.trainPcn(trainSet, trainLabels, tipeOfTrain)

    # avalia o classificador utilizando o conjunto de teste
    O = p.testPcn(testSet)
    O = Perceptron.binaryToDecimal(O)

    print('Acurácia', Perceptron.calcAccuracy(O, testLabels), '%')

    conf_mat = confusion_matrix(testLabels, O)
    print('Matriz de confusão\n', conf_mat)

    # plota a matriz de confusão
    sns.heatmap(conf_mat, annot=True, cmap='Blues')
    plt.show()
    #sns.heatmap(conf_mat / np.sum(conf_mat), annot=True, fmt='.2%', cmap='Blues')
    #plt.show()


if __name__ == "__main__":
    main()