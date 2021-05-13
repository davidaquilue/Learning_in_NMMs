''' Some test per no anar repetint tot bro'''
import numpy as np
import matplotlib.pyplot as plt
from signals import build_p_inputs, build_dataset, add_shift
from matfuns import fastcrosscorrelation, findlayer

inputnodes = 3
t = np.arange(0, 40, 0.001)
offset = 80
n = 20
corrpairs = ((0, 1), (0, 2), (1, 2))

tuplenetwork = (3,3,3)
Nnodes = 0
for layernodes in tuplenetwork:
    Nnodes += layernodes
# Let us test the findlayer function
for ii in range(Nnodes):
    print(ii)
    print(findlayer(ii, tuplenetwork))

# Okei comprovem que sí que funciona i que per tant es pot utilitzar dintre del
# codi principal de integració de les dinàmiques.


'''
p_inputs = build_p_inputs(inputnodes, t, 80, corrpairs[1])
print(p_inputs.shape)
p_inputs = add_shift(p_inputs, 5, 0.001, corrpairs[1])
print(p_inputs.shape)

fig, axes = plt.subplots(3, 1)

for ii in range(inputnodes):
    ax = axes[ii]
    ax.plot(t, p_inputs[ii])

plt.show()

dataset = build_dataset(n, inputnodes, corrpairs, t, offset)
print(len(dataset))
print(dataset[0].shape)
fig, axes = plt.subplots(3, 3)

# It is clear that each element of the dataset is the same
print(np.linalg.norm(dataset[1]-dataset[2]))

for jj in range(len(corrpairs)):
    data_set = dataset[jj]
    for ii in range(inputnodes):
        ax = axes[ii, jj]
        ax.plot(t, data_set[12, ii])

    print('cc1: ' + str(fastcrosscorrelation(data_set[12,0], data_set[12,1])))
    print('cc2: ' + str(fastcrosscorrelation(data_set[12,0], data_set[12,2])))
    print('cc3: ' + str(fastcrosscorrelation(data_set[12,2], data_set[12,1])))
plt.show()
'''


