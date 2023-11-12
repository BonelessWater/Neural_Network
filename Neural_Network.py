import math
import random
import matplotlib.pyplot as plt
import numpy as np

def summation(array, node_pre_layer):
    sum = 0
    for i in range(node_pre_layer):
        sum += array[i][0] * array[i][1]
    
    return sum

def sigmoid(x):
    return 1 / (1 + math.pow(math.e, -x))

def input_function(random_input):
    return math.sin( math.pow(random_input[0], 2)) * math.cos(random_input[0])

class BaseNetwork:
    def __init__(self, data, network_structure, best_weight_array):
        self.data = data
        self.network_structure = network_structure
        self.network = []
        self.weight_array = best_weight_array

    # Appends number of weights according to network structure and appends random values
    def initweights(self):
        if self.weight_array == []:
            total_weights = 0
            for i in range (len(self.network_structure) - 1):
                total_weights += self.network_structure[i] * self.network_structure[i+1]
            for i in range(total_weights):
                self.weight_array.append(random.uniform(0, 1))

    def randomize_weights(self):
        for i in range (len(self.weight_array)):
            self.weight_array[i] += random.uniform(-0.5, 0.5)
            if self.weight_array[i] > 1:
                self.weight_array[i] = 1
            elif self.weight_array[i] < 0:
                self.weight_array[i] = 0

    def modify_network(self):
        random.seed()
        layer_prob = random.uniform(0, 1)
        node_prob = random.uniform(0, 1)
        remove_node_prob = random.uniform(0, 1)

        # Adding a new layer
        if layer_prob > 0.9:
            self.network_structure.insert( (len(self.network_structure) - 1), 1)

        # Adding a new node
        if node_prob > 0.5 and self.weight_array != [] and self.network != []:
            self.network_structure[len(self.network_structure) - 1] + 1
            # Adds edges according to the number of nodes in the 3nd-to-last and last layer
            self.network[-2].append([])
            for i in range (self.network_structure[-3]):
                self.network[-2][-1].append([0, random.uniform(0, 1)])
            self.network[-2][-1].append(0)

        # Removing a node


    def initlayers(self):

        # Appends structure, input and empty array for each layer
        self.network.append(self.network_structure)
        self.network.append(self.data)

        for i in range(len(self.network_structure) - 1):
            self.network.append([])

        counter = 0

        # Sets weights of the network according to the weight_array
        for i in range (1, len(self.network_structure)):

            nodes_layer = self.network_structure[i]
            nodes_pre_layer = self.network_structure[i - 1]

            for j in range (nodes_layer):
                self.network[i + 1].append([])

                for k in range (nodes_pre_layer):
                    self.network[i + 1][j].append([0, self.weight_array[counter]])
                self.network[i + 1][j].append(0)  

# Takes values from previous layer and adds it to the "signal" of the next layer at every node
        for layer in range(2, len(self.network_structure) + 1):
            for node in range(self.network_structure[layer - 1]):
                nodes_pre_layer = self.network_structure[layer - 2]
                for edge in range(nodes_pre_layer):
                    
# The first layer will only contain float values
                    if layer == 2:
                        self.network[2][node][edge][0] = self.network[1][edge]

# The rest of the layers will contain nodes with signals and weights inside them
                    if layer != 2:
                        self.network[layer][node][edge][0] = self.network[layer - 1][edge][-1]
        
# Summation of signal times weight for each edge in a node and applies sigmoid function
                self.network[layer][node][-1] = sigmoid(summation(self.network[layer][node], nodes_pre_layer))
        print(self.network)


    def return_weights(self):
        return self.weight_array

    def return_difference(self):
        x = self.network[-1][0][-1]
        return ( math.pow(x , 2) - 
                    math.pow(input_function(self.data), 2) )

def main(population, generation):

    # Input, Hidden layers, Output
    network_structure = [1, 7, 6, 4, 1]

    best_difference = 1
    generations = []
    differences = []
    best_weight_array = []

    for i in range (generation):
        
        generations.append(i)
        differences.append(abs(best_difference))

        for j in range (population):

            random_input = [random.uniform(-30, 30)]

            newNetwork = BaseNetwork(random_input, network_structure, best_weight_array)

            newNetwork.initweights()
            newNetwork.randomize_weights()

            #newNetwork.modify_network()
            newNetwork.initlayers()



            if abs(newNetwork.return_difference()) < best_difference:
                best_difference = newNetwork.return_difference()
            best_weight_array = newNetwork.return_weights()
        
# Plot data
   
    plt.style.use('_mpl-gallery')
    
    x = np.array(generations)
    y = np.array(differences)

    # print(differences)
    # print(generations)

    plt.scatter(x, y)
    plt.axhline(y = 0, color = 'r', linestyle = '-') 
    plt.show()

main(2, 50)

# Have the code randomly alter the amount of nodes in the structure and rememebr to give it a random starting weight.
# The structure must also be taken into acount when logging the "best" network. AKA weights and structure now become important.
