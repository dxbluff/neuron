import numpy as np
 
def sigmoid(x):
    # Наша функция активации: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))
 
 
class OurNeuralNetwork:
    def __init__(self):
        weights = np.array([0,1])
        bias = 0
        
class Neuron:
    def __init__(self, weights, bias):
      self.weights = weights
      self.bias = bias        
        
    def feedforward(self,inputs):        
        total=np.dot(self.weights,inputs)+ self.bias
        return sigmoid(total)

        self.h1=Neuron(weights,bias)

        self.h2=Neuron(weights,bias)     
        self.o1=Neuron(weights,bias)
     

    def feedforward  (self, x):
      out_h1 = self.h1.feedforward(x)
      out_h2 = self.h2.feedforward(x)
 
        # Вводы для о1 являются выводами h1 и h2
      out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
 
      return out_o1

network = OurNeuralNetwor()                           
x = np.array([2, 3])
print(network.feedforward(x))  # 0.7216325609518421
