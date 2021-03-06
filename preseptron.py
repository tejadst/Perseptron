import numpy as np

_training_data = np.array(
    [[0.05, 0.02], [0.09, 0.11], [0.12, 0.20], [0.15, 0.22], [0.2, 0.25], [0.75, 0.75], [0.8, 0.83], [0.82, 0.8],
     [0.9, 0.89], [0.95, 0.89]])

_test_data = np.array(
    [[0.09, 0.04], [0.1, 0.1], [0.14, 0.21], [0.18, 0.24], [0.22, 0.28], [0.77, 0.78], [0.79, 0.81], [0.84, 0.82],
     [0.94, 0.93], [0.98, 0.99]])

_output = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

weights_I1 = np.array([[0.5, 0.4, 0.1], [0.2, 0.6, 0.2]])
weights_12 = np.array([[0.1, 0.55, 0.35], [0.2, 0.45, 0.35], [0.25, 0.1, 0.6]])
weights_2O = np.array([[0.3, 0.35], [0.35, 0.25], [0.45, 0.2]])

error_rate = 0.3


class NeuralNetwork:

    def segmoid(self,x):
        return 1/(1+np.exp(-x))

    def activation(self,layer,weights):
        return self.segmoid(np.matmul(layer,weights))

    def train(self,training_data,output):
        global weights_I1
        global weights_12
        global weights_2O
        for i in range(10000):
            _input = training_data[i%10,:]
            layer1 =self.activation(_input,weights_I1)
            layer2 = self.activation(layer1,weights_12)
            output_layer = self.activation(layer2,weights_2O)
            error = output[i%10,:]-output_layer
            error_layer2=layer2*(1-layer2)*np.matmul(error,weights_2O.T)
            error_layer1 = layer1 * (1 - layer1) * np.matmul(error_layer2, weights_12.T)
            weights_2O = weights_2O + error_rate * np.repeat(error.reshape(1,2),3,axis=0) * np.repeat(layer2.reshape(3,1),2,axis=1)
            weights_12 = weights_12 + error_rate * np.repeat(error_layer2.reshape(1,3),3,axis=0) * np.repeat(layer1.reshape(3,1),3,axis=1)
            weights_I1 = weights_I1 + error_rate * np.repeat(error_layer1.reshape(1,3),2,axis=0) * np.repeat(_input.reshape(2,1),3,axis=1)

    def test(self,test_data):
        global weights_I1
        global weights_12
        global weights_2O
        for i in range(test_data[:, 0].size):
            _input = test_data[i, :]
            layer1 = self.activation(_input, weights_I1)
            layer2 = self.activation(layer1, weights_12)
            output_layer = self.activation(layer2, weights_2O)
            print(output_layer)

    def test_case(self,_input):
        global weights_I1
        global weights_12
        global weights_2O
        layer1 = self.activation(_input, weights_I1)
        layer2 = self.activation(layer1, weights_12)
        output_layer = self.activation(layer2, weights_2O)
        print(output_layer)


if __name__ == '__main__':
    AI=NeuralNetwork()
    AI.train(_training_data,_output)
    while 1:
        input_=np.array([float(input('enter input\n')),float(input())])
        AI.test_case(input_)
