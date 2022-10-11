import numpy as np
import math
from sklearn.metrics import mean_absolute_error
from random import randint


class LogisticRegres:
    def __init__(self, weights=None, count_weights=9):
        if weights is None:
            self.weights = np.array([1] * count_weights)
        else:
            self.weights = np.array(weights)
        self.error = 1.00
        self.count_weights = len(self.weights)

    # логистическая функция/сигмоид
    def sigmoid(self, x):
        z = self.weights[0]
        for i in range(len(x)):
            z += x[i] * self.weights[i + 1]
        return 1 / (1 + math.exp(-z))

    # возвращает вероятность 1
    def predict(self, x):
        return self.sigmoid(x)

    # возвращает значение 1 или 0
    def pred(self, x):
        return round(self.predict(x))

    # градиентный спуск, максимальное правдоподобие
    def _gradient_descent_step(self, x, y, alpha=0.0001):

        b = [randint(1,100)/100 for i in range (self.count_weights)]
        for i in range(len(y)):
            b[0] += y[i] - self.predict(x[i])
            for j in range(1, self.count_weights):
                b[j] += (y[i] - self.predict(x[i])) * x[i][j - 1]
        arr = []
        for i in range(self.count_weights):
            arr.append(self.weights[i] + alpha * b[i])
        return arr

    # обучение классификатора
    def learn(self, x, y, x_test, y_test, error=mean_absolute_error, log_print=False):
        self.graph = []
        i = 0
        self.error, last_error, need_weights = 100.0, 100.0, []
        print("Learning...")
        while self.error:
            self.weights = self._gradient_descent_step(x, y)
            p = np.array([self.pred(i) for i in x_test])
            e = error(p, y_test)
            self.graph.append(e)
            if log_print:
                print('Iteration:\t', i + 1,   '\tError:\t', e, '\tWeights\t', self.weights)
            if self.error > e:
                self.error = e
                need_weights = self.weights
            else:
                last_error = e
                if i > 1500 and abs(e - last_error) < 0.001:
                    break
            i += 1
        self.weights = need_weights
        print(f'The learning took {i+1} iterations')


