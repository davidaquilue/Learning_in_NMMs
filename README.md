# Learning_in_NMMs
Code developed for my Bachelor's degree Thesis titled Learning in Neural Mass Models.

The main objective of the Bachelor's Thesis was to build a model representing a network of interconnected cortical columns that could learn a certain pattern of activity. Thus, I built a feedforward network of Jansen-Rit (JR) models.

One JR model represents the averaged activity of the interaction between three populations of neurons in the cortex (pyramidal cells, excitatory interneurons and inhibitory interneurons). Interactions between JR models (also can be called columns or nodes in this network model) are determined by a set of weights. I applied a Genetic Algorithm (GA) to the set of weights in order to find which values of the weights resulted in a desired set of activities.

The set of activities that I considered can be divided in three situations. In each situation, two nodes in the first layer of the network have correlated activities between them (they receive the same input signal) while the third one has no special correlation with the other two (it receives a random signal, unrelated to the signal that the other two receive). The objective is to have, in the output layer, a high correlation between the two models parallel to the correlated models in the input layer while the third node is as uncorrelated as possible. The following figure illustrates this situations:
![alt text](https://github.com/davidaquilue/Learning_in_NMMs/blob/main/Results/situations.png)

For this, a fitness function has to be defined so that the algorithm knows what it needs to optimize.

I built the entire model from scratch, making use of libraries like numpy and numba for simulating the network of models and DEAP for the implementation of the Genetic Algorithm (amongst others).

My Bachelor's Thesis can be found in:
https://upcommons.upc.edu/handle/2117/355315

All the functions developed for the Bachelor's Thesis are found in the main directory. An additional test on Hebbian-like learning network of Neural Mass Models, inspired by Filippo Cona and Mauro Ursino can be found in /Extras/Hebbian. In /Results/ one can see some figures of the evolution of the GA, the optimal set of weights found, the resulting dynamics of the model, and the correlations, amongst others. However, I recommend skimming through my thesis for a more detailed and ordered description of methods and results.
