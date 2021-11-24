import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SigmoidNeuralNetwork:
    def __init__(self, shape, learningrate=0.01) -> None:
        self.shape = shape
        self.lr = learningrate
        self.weights = []
        self.biases = []
        for i, size in enumerate(shape[1:]):
            self.weights.append(np.random.normal(0, pow(size, -0.5), (size, shape[i])))
            self.biases.append(np.zeros((size, 1)))
        
    def predict(self, x, last_only=True):
        _x = np.array(x, ndmin=2).T
        expected = [_x]
        for i in range(len(self.weights)):
            expected.append(sigmoid((self.weights[i] @ expected[i]) + self.biases[i]))
        return expected[-1] if last_only else expected
    
    def train(self, x, y):

        expected = self.predict(x, last_only=False)

        error = y - expected[-1]
        for i in range(len(self.weights) - 1, -1 , -1):
            error *= expected[i+1] * (1 - expected[i+1])
            current_error = (self.weights[i].T @ error)
            self.weights[i] += self.lr * error @ expected[i].T
            self.biases[i] += self.lr * error
            error = current_error