'''
Trabalho 2 - Sistemas Inteligentes
Perceptron - Classificação de Distribuições Gaussianas
Sarah R. L. Carneiro
'''
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:

    def __init__ (self, learningRate, numberOfTimes, weights):
        
        self.learningRate = learningRate # taxa de apredisagem
        self.numberOfTimes = numberOfTimes # numero de epocas
        self.W = weights # pesos sinapticos
        self.activationFunc = self.unitStep # funcao de ativacao

    def testPcn (self, X):
        return self.predict(X)

    def trainPcn (self, X, T, tipeOfTrain):
        
        # treinamento de batch
        if (tipeOfTrain == 0):

            for i in range(self.numberOfTimes):
                
                O = self.predict(X)

                # W = W + lambda * X * (T - O)
                self.W += self.learningRate * np.dot(X.T, (T - O)) 
            
        # treinamento sequencial
        else:

            for i in range(self.numberOfTimes):

                for k, x_k in enumerate(X):
                    
                    O = self.predict(X)

                    # w_ij = w_ij + lambda * x_ki * (t_kj - o_kj)
                    self.W += self.learningRate * np.dot(x_k[np.newaxis].T, (T[k,:] - O[k,:])[np.newaxis])
                         
    # f(XW - b)
    def predict (self, X):

        h = np.dot(X, self.W)
        return self.activationFunc(h)

    # funcao de ativacao - degrau unitario
    def unitStep (self, x):
        
        return np.where(x>0, 1, 0)

def main():
    
    print('\n\n===== Perceptron - Classificação de Distribuições Gaussianas =====\n\n')

    learningRate = float(input('Insira a taxa de aprendizagem:'))
    numberOfTimes = int(input('Insira o número de épocas:'))
    N = int(input('Insira o número de padrões por classe:'))
    percentTrain = float(input('Insira a porcentagem de dados que serão utililizados no treinamento:'))
    tipeOfTrain = int(input('Insira [0] para treinamento de lote e [1] treinamento sequencial:'))
    
    if (tipeOfTrain != 0 and tipeOfTrain != 1):
        print('Entrada inválida!!')
        exit(0)
    
    for i in range(0, 3):

        # valor medio de cada gaussiana
        average = float(input('Insira o valor medio da {}° gaussiana: '.format(i+1)))

        # desvio padrao de cada gaussiana
        sd = float(input('Insira o valor do desvio padrão da {}° gaussiana: '.format(i+1)))
        
        # padroes de entrada
        if (i == 0):
            X = np.random.normal(average, sd, (N, 2))
        else:
            X = np.concatenate((np.random.normal(average, sd, (N, 2)), X), axis=0)
    
    # adiciona o bias aos padroes de entrada
    X = np.concatenate((np.full((3*N, 1), -1.0), X), axis=1)

    # cria os conjuntos de treinamento e de teste
    rand_state = np.random.RandomState(0)
    rand_state.shuffle(X)
    idx = int(percentTrain*N*3)
    trainSet = X[0:idx, :]
    testSet = X[idx+1:len(X), :]

    # alvos
    T = np.zeros((N,2))
    T = np.concatenate((T, np.concatenate((np.zeros((N,1)), np.ones((N,1))), axis=1)), axis=0)
    T = np.concatenate((T, np.concatenate((np.ones((N,1)), np.ones((N,1))), axis=1)), axis=0)
    rand_state.seed(0)
    rand_state.shuffle(T)
    tTrain = T[0:idx, :]
    tTest = T[idx+1:len(X), :]


    # pesos sinapticos
    weights = np.random.normal(0, 0.01, (3, 2))

    p = Perceptron (learningRate= learningRate, numberOfTimes= numberOfTimes, weights = weights)

    # plota os padroes de entrada por classe (conjunto de treinamento)
    fig = plt.figure()
    plt.scatter(trainSet[:,1], trainSet[:,2],marker = 'o', c = (tTrain[:,0] + tTrain[:,1]))

    # treina o perceptron
    p.trainPcn(trainSet, tTrain, tipeOfTrain)

    # plota as fronteiras entre as classes
    colors = ['red', 'black']
    for j in range(0, 2):
        x_plot = np.linspace(-4, 4, 50)
        plt.plot(x_plot, (p.W[0][j] - (p.W[1][j] * x_plot))/p.W[2][j], '--', color=colors[j])
    
    plt.title('Conjunto de treinamento')
    plt.show()

    # plota os padroes de entrada por classe (conjunto de teste)
    fig = plt.figure()
    plt.scatter(testSet[:,1], testSet[:,2],marker = 'o', c = (tTest[:,0] + tTest[:,1]))

    # avalia o classificador utilizando o conjunto de teste

    O = p.testPcn(testSet)
    print('Acurácia', (1 - np.sum(np.absolute(O-tTest))/len(tTest)) * 100, '%')

    # plota as fronteiras entre as classes
    colors = ['red', 'black']
    for j in range(0, 2):
        x_plot = np.linspace(-4, 4, 50)
        plt.plot(x_plot, (p.W[0][j] - (p.W[1][j] * x_plot))/p.W[2][j], '--', color=colors[j])
    
    plt.title('Conjunto de teste')
    plt.show()

if __name__ == '__main__':
    main()