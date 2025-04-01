"""
Aiming a Dynaimic Graph-structured NeuronNetwork
However this is only a naive implementation of SGD, with potentials to be a graph-like structure
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from collections import deque

"""
Activation function in this case in tanh
Loss is the Euclidean loss
"""
class Model:
  class Neuron:
    def __init__(self, name, prev, next):
      self.name, self.prev, self.next = name, prev, next
      self.bias = np.random.uniform(-0.1, 0.1)

  def __init__(self, input_size, output_size):
    self.Input_layer = [Model.Neuron(f"input{i}", [], []) for i in range(input_size)]
    self.Output_layer = [Model.Neuron(f"output{o}", [], []) for o in range(output_size)]
    for i in range(input_size): self.Input_layer[i].next = self.Output_layer
    for o in range(output_size): self.Output_layer[o].prev = self.Input_layer
    self.weight = {}
    for u in self.Input_layer:
      for v in self.Output_layer:
        self.weight[(u,v)] = np.random.uniform(-0.1, 0.1)
    self.all_neurons = self.Input_layer + self.Output_layer

  def forward(self, X, batch_size):
    assert X.shape == (batch_size,len(self.Input_layer))
    a = {q: np.zeros(batch_size) for q in self.all_neurons}

    for i, n in enumerate(self.Input_layer):
      a[n] = X[:, i]

    q = deque()
    for i in self.Input_layer:
      q.append(i)

    cnt = {q: 0 for q in self.all_neurons}

    while len(q) != 0:
      c = q.popleft()
      a[c] = np.tanh(a[c] + c.bias)
      for n in c.next:
        a[n] = a[n] + a[c] * self.weight[(c,n)]
        cnt[n] += 1
        if cnt[n] == len(n.prev):
          q.append(n)
    return a

  def eval(self, X):
    a = self.forward(X, len(X))
    return np.array([a[o] for o in self.Output_layer]).T

  def backward(self, X, Y, batch_size, learning_rate):
    assert X.shape == (batch_size,len(self.Input_layer))
    assert Y.shape == (batch_size,len(self.Output_layer))
    a = self.forward(X, batch_size)
    par_a = {q: np.zeros(batch_size) for q in self.all_neurons}
    for o, n in enumerate(self.Output_layer):
      par_a[n] = 2 * (a[n] - Y[:, o])

    q = deque()
    for o in self.Output_layer:
      q.append(o)

    cnt = {q: 0 for q in self.all_neurons}

    while len(q) != 0:
      c = q.popleft()
      par_b = par_a[c] * (1-a[c]**2)
      c.bias -= np.mean(par_b) * learning_rate
      for p in c.prev:
        par_a[p] += par_a[c] * (1-a[c]**2) * self.weight[(p,c)]
        par_w_pc = par_a[c] * (1-a[c]**2) * a[p]
        self.weight[(p,c)] -= np.mean(par_w_pc) * learning_rate
        cnt[p] += 1
        if cnt[p] == len(p.next):
          q.append(p)

  def addLayer(self, mid_size, UP, DOWN):
    Mid_layer = [Model.Neuron(f"mid{o}", [], []) for o in range(mid_size)]
    for m, mid in enumerate(Mid_layer):
      Mid_layer[m].prev = UP
      for u in UP:
        self.weight[(u,mid)] = np.random.uniform(-0.1, 0.1)
      Mid_layer[m].next = DOWN
      for v in DOWN:
        self.weight[(mid,v)] = np.random.uniform(-0.1, 0.1)
    for u in UP:
      u.next = Mid_layer
    for v in DOWN:
      v.prev = Mid_layer
    for u in UP:
      for v in DOWN:
        self.weight.pop((u, v))
    self.all_neurons += Mid_layer
    return Mid_layer

  def train(self, X, Y, batch_size, epochs, learning_rate):
    l = len(X)
    for epoch in range(epochs):
      data=[(X[_], Y[_]) for _ in range(len(X))]
      random.shuffle(data)
      for _ in range(len(X)):
        X[_],Y[_]=data[_]
      loss = 0
      for batch in range(int(l / batch_size)):
        L, R = batch * batch_size, (batch + 1) * batch_size
        x_train, y_train = X[L:R], Y[L:R]
        self.backward(x_train, y_train, batch_size, learning_rate)
        output = self.eval(x_train)
        loss += np.sum(((y_train - output) ** 2), axis=(0,1))
      loss = ((loss) ** 0.5) / (int(l / batch_size) * batch_size)
      print(f"Epoch {epoch}/{epochs}, Loss:{loss}")


"""
A small test on MNIST is included below
"""
import tensorflow as tf
tf.random.set_seed(42)
# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Flatten images to 1D vector of 784 features (28*28)
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

def test(model, X, Y, batch_size):
  k = int(len(X)/batch_size)
  for i in range(k):
    Y_hat=model.eval(X[i*batch_size:(i+1)*batch_size])
    wrong=0
    for j in range(batch_size):
      max1,max2,id1,id2=-999,-999,-1,-1
      for l in range(10):
        if max1 < Y_hat[j][l]:
          max1,id1=Y_hat[j][l],l
        if max2 < Y[i*batch_size+j][l]:
          max2,id2=Y[i*batch_size+j][l],l
      if id1 != id2: wrong+=1
    print(f"batch: {i}, accuracy: {(batch_size-wrong)/batch_size*100}%")
mod2 = Model(784, 10)
mod2.addLayer(32, mod2.Input_layer, mod2.Output_layer)
print(len(mod2.Input_layer))
mod2.train(x_train, y_train, 512, 10, 0.01)
test(mod2, x_test, y_test, 500)
