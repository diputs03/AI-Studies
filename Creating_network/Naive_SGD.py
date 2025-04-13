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
  def __init__(self, input_size, output_size):
    self.idcnt = 0
    self.prev, self.next = {}, {}
    self.neurons = set()

    self.Input_layer = [self.idcnt+i for i in range(input_size)]
    self.neurons.update([self.idcnt+i for i in range(input_size)])
    self.idcnt+=input_size

    self.Output_layer = [self.idcnt+o for o in range(output_size)]
    self.neurons.update([self.idcnt+o for o in range(output_size)])
    self.idcnt+=output_size

    for i in self.Input_layer: self.next[i], self.prev[i] = self.Output_layer.copy(), []
    for o in self.Output_layer: self.prev[o], self.next[o] = self.Input_layer.copy(), []

    self.weight = {}
    self.weight_gsum, self.weight_gsqr = {}, {}
    for u in self.Input_layer:
      for v in self.Output_layer:
        self.weight[(u,v)] = np.random.uniform(-0.1, 0.1)
        self.weight_gsum[(u,v)], self.weight_gsqr[(u,v)] = 0, 0

    self.bias = {}
    self.bias_gsum, self.bias_gsqr = {}, {}
    for i in self.neurons:
      self.bias[i] = np.random.uniform(-0.1, 0.1)
      self.bias_gsum[i], self.bias_gsqr[i] = 0, 0

  def addLayer(self, mid_size, UP, DOWN):
    Mid_layer = [self.idcnt+m for m in range(mid_size)]
    self.neurons.update([self.idcnt+m for m in range(mid_size)])
    self.idcnt+=mid_size

    for m in Mid_layer:
      self.bias[m] = np.random.uniform(-0.1, 0.1)
      self.bias_gsum[m], self.bias_gsqr[m] = 0, 0

      self.prev[m] = UP.copy()
      for u in UP:
        self.weight[(u,m)] = np.random.uniform(-0.1, 0.1)
        self.weight_gsum[(u,m)], self.weight_gsqr[(u,m)] = 0, 0

      self.next[m] = DOWN.copy()
      for v in DOWN:
        self.weight[(m,v)] = np.random.uniform(-0.1, 0.1)
        self.weight_gsum[(m,v)], self.weight_gsqr[(m,v)] = 0, 0

    for u in UP:
      self.next[u] = Mid_layer.copy()
    for v in DOWN:
      self.prev[v] = Mid_layer.copy()

    for u in UP:
      for v in DOWN:
        self.weight.pop((u,v))
        self.weight_gsum.pop((u,v))
        self.weight_gsqr.pop((u,v))
    return Mid_layer

  def addNode(self, u, v):
    n = self.idcnt
    self.idcnt += 1
    self.neurons.add(n)
    self.next[n], self.prev[n] = [], []
    self.bias[n] = np.random.uniform(-0.1, 0.1)
    self.bias_gsum[n], self.bias_gsqr[n] = 0, 0

    self.next[u].append(n)
    self.prev[n].append(u)
    self.next[n].append(v)
    self.prev[v].append(n)
    self.weight[(u,n)] = np.random.uniform(-0.1, 0.1)
    self.weight_gsum[(u,n)], self.weight_gsqr[(u,n)] = 0, 0
    self.weight[(n,v)] = np.random.uniform(-0.1, 0.1)
    self.weight_gsum[(n,v)], self.weight_gsqr[(n,v)] = 0, 0

  def __forward(self, X, batch_size):
    assert X.shape == (batch_size,len(self.Input_layer)), \
      f"X.shape={X.shape}, where {(batch_size,len(self.Input_layer))} is expected"
    a = {q: np.zeros(batch_size) for q in self.neurons}

    for i, n in enumerate(self.Input_layer):
      a[n] = X[:, i].copy()

    q = deque()
    for i in self.Input_layer:
      q.append(i)

    cnt = {q: 0 for q in self.neurons}

    while len(q) != 0:
      c = q.popleft()
      a[c] = np.tanh(a[c] + self.bias[c])
      for n in self.next[c]:
        a[n] = a[n] + a[c] * self.weight[(c,n)]
        cnt[n] += 1
        if cnt[n] == len(self.prev[n]):
          q.append(n)
    return a

  def evaluate(self, X):
    a = self.__forward(X, len(X))
    return np.array([a[o] for o in self.Output_layer]).T

  def __backward(self, X, Y, batch_size, learning_rate, dsum, dsqr):
    assert X.shape == (batch_size,len(self.Input_layer)), \
      f"X.shape={X.shape}, where {(batch_size,len(self.Input_layer))} is expected"
    assert Y.shape == (batch_size,len(self.Output_layer)), \
      f"X.shape={Y.shape}, where {(batch_size,len(self.Output_layer))} is expected"
    a = self.__forward(X, batch_size)

    db, dw = {}, {}

    par_a = {q: np.zeros(batch_size) for q in self.neurons}
    for o, n in enumerate(self.Output_layer):
      par_a[n] = 2 * (a[n] - Y[:, o])

    q = deque()
    for o in self.Output_layer:
      q.append(o)

    cnt = {q: 0 for q in self.neurons}

    while len(q) != 0:
      c = q.popleft()
      par_b = par_a[c] * (1-a[c]**2)

      gbias = par_b
      self.bias_gsum[c] = (1-dsum)*np.sum(gbias)/batch_size + dsum*self.bias_gsum[c]
      self.bias_gsqr[c] = (1-dsqr)*np.sum(gbias**2)/batch_size + dsqr*self.bias_gsqr[c]
      db[c] = -learning_rate * self.bias_gsum[c] / (self.bias_gsqr[c]**(1/2)+1)

      for p in self.prev[c]:
        par_a[p] += par_a[c] * (1-a[c]**2) * self.weight[(p,c)]
        gweight = par_a[c] * (1-a[c]**2) * a[p]
        self.weight_gsum[(p,c)] = \
         (1-dsum)*np.sum(gweight)/batch_size + dsum*self.weight_gsum[(p,c)]
        self.weight_gsqr[(p,c)] = \
         (1-dsqr)*np.sum(gweight**2)/batch_size + dsqr*self.weight_gsqr[(p,c)]
        dw[(p,c)] = \
         -learning_rate * self.weight_gsum[(p,c)] / (self.weight_gsqr[(p,c)]**(1/2)+1)

        cnt[p] += 1
        if cnt[p] == len(self.next[p]):
          q.append(p)

    return dw, db

  def train(self, x, y, batch_size, epochs, learning_rate, verbose=True):
    assert len(x) == len(y)
    l = len(x)
    Loss = [0 for epoch in range(epochs)]
    for epoch in range(epochs):
      X, Y = x.copy(), y.copy()
      data=[(X[_], Y[_]) for _ in range(l)]
      random.shuffle(data)
      for _ in range(l):
        X[_],Y[_]=data[_]
      loss = 0
      for batch in range(int(l / batch_size)):
        L, R = batch * batch_size, (batch + 1) * batch_size
        x_split, y_split = X[L:R], Y[L:R]
        dw, db = self.__backward(
          x_split, y_split, batch_size, learning_rate,
          .9, .9
        )
        for w in self.weight:
            self.weight[w] += dw[w]
        for p in self.neurons:
            self.bias[p] += db[p]
        output = self.evaluate(x_split)
        loss += np.sum(((y_split-output) ** 2), axis=(0,1))
      loss = ((loss) ** 0.5) / (int(l / batch_size) * batch_size)
      if verbose:
        print(f"Epoch {epoch}/{epochs}, Loss:{loss}")
      Loss[epoch] = loss
    fig, ax = plt.subplots()
    ax.plot([i for i in range(epochs)],Loss,'+',linewidth=2)
    return fig, ax


def test(model, X, Y, batch_size):
  k = int(len(X)/batch_size)
  for i in range(k):
    Y_hat=model.evaluate(X[i*batch_size:(i+1)*batch_size])
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

mod2 = Model(784, 10)
mod2.addLayer(32, mod2.Input_layer, mod2.Output_layer)
print(len(mod2.Input_layer))
mod2.train(x_train, y_train, 512, 10, 0.01)
test(mod2, x_test, y_test, 10000)
