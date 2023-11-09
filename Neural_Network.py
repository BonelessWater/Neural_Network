import math

def summation(array, nodeperlayer):
    sum = 0
    for i in range(nodeperlayer):
        sum += array[i][0] * array[i][1]
    
    return sum

def sigmoid(x):
    return 1 / (1 + math.pow(math.e, -x))

layer0 = [0.5, 0.6, 0.7, 0.8]

def input_function(layer0):
    return layer0[0] - layer0[3] + math.pow(layer0[1], 2) - (layer0[2]/2)

class BaseNetwork:
    def __init__(self, data, inputs, outputs, layers, nodeperlayer):
        self.data = data
        self.inputs = inputs
        self.outputs = outputs
        self.layers = layers
        self.nodeperlayer = nodeperlayer
        self.network = [data]
    
    def initlayers(self):

        for i in range(self.layers - 1):
            self.network.append([])

        for i in range(1, self.layers - 1):
            for j in range(self.nodeperlayer):
                self.network[i].append([[0, 0.5], [0, 0.5], [0, 0.5], [0, 0.5], 0])
        self.network[self.layers - 1].append([[0, 0.5], [0, 0.5], [0, 0.5], [0, 0.5], 0])
# Takes values from previous layer and adds it to the "signal" of the next layer at every node
        for layer in range(1, self.layers - 1):
            for node in range(self.nodeperlayer):
                for edge in range(self.nodeperlayer):
                    
# The first layer will only contain float values
                    if layer == 1:
                        self.network[layer][node][edge][0] = self.network[layer - 1][node]
# The rest of the layers will contain nodes with signals and weights inside them
                    if layer != 1:
                        self.network[layer][node][edge][0] = self.network[layer - 1][node][-1]

# Summation of signal times weight for each edge in a node and applies sigmoid function
            for node in range(self.nodeperlayer):
                # print(self.network[layer])
                self.network[layer][node][-1] = sigmoid(summation(self.network[layer][node], self.nodeperlayer))

# Summation of final layer              
        for i in range(self.nodeperlayer):
            self.network[-1][0][i][0] = self.network[-2][i][-1]
        self.network[-1][0][-1] = sigmoid(summation(self.network[-1][0], self.nodeperlayer))



# Parameter: input, # of inputs, # of outputs, # of layers , # of nodes per layer
myNetwork = BaseNetwork(layer0, 4, 1, 4, 4)

myNetwork.initlayers()



# By creating hundreds of iterations of mutated neural networks and keeping the most successful, 
# the network should learn on its own

