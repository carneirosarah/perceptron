'''
Trabalho 1 - Sistemas Inteligentes
Treinamento de um perceptron que simula o funcionamento de uma porta lógica OR
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
    
    def fit (self, X, T):

        fig = plt.figure()
        plt.scatter(X[:,1], X[:,2],marker = 'o', c = T)
            
        for i in range(self.numberOfTimes):

            print(i, 'época')
            O = self.predict(X)
            print(O)

            for k, x_k in enumerate(X):

                # w_ij = w_ij + lambda * x_ki * (t_kj - o_kj)
                self.W += self.learningRate * (T[k] - O[k]) * x_k
                print (k, x_k, self.W)

            x_plot = np.linspace(-0.5, 1.5, 50)
            print(x_plot)
            line = (self.W[0] - (self.W[1] * x_plot))/self.W[2]
            print('LINE', line)
            plt.plot(line, x_plot)
            plt.show()
                
            
    # f(XW - b)
    def predict (self, X):

        h = np.dot(X, self.W)
        return self.activationFunc(h)

    # funcao de ativacao - degrau unitario
    def unitStep (self, x):
        
        return np.where(x>0, 1, 0)

def main():
    
    X = np.array([[-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [-1, 1,1]]) # padroes de treinamento
    T = np.array([0, 1, 1, 1]) # alvos

    print('\n\n===== Perceptron - Porta OR =====\n\n')

    learningRate = float(input('Insira a taxa de aprendizagem:'))
    numberOfTimes = int(input('Insira o número de épocas:'))
    flag = input('Você deseja gerar os pesos sinápticos de forma aleátoria? s ou n: ')

    if (flag == 's'):
        
        weights = np.random.normal(0,0.5,3)
        print(weights)

    elif (flag == 'n'):
        
        weights = np.empty(3)

        for i in range(0, 3):
            
            weights[i] = float(input('Insira o {}° elemento: '.format(i+1)))
        
        print(weights)

    else:
        print('Entrada inválida!!')
        exit(0)    
    
    p = Perceptron (learningRate= learningRate, numberOfTimes= numberOfTimes, weights = weights)
    p.fit(X, T)

if __name__ == '__main__':
    main()
