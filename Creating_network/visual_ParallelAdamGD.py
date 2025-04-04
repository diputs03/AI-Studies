#@title Aiming a Dynaimic Graph-structured NeuronNetwork
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor

"""
Activation function in this case in \tanh, thus
\dfrac{d\tanh(x)}{dx}=1-\tanh^2(x)
however, for other activation funtions
\dfrac{d\sigma(x)}{dx}=\sigma(x)\cdot\left\big(1-\sigma(x)\right\big)
\dfrac{d\mathop{\mathrm{ReLu}}(x)}{dx}=\begin{cases}1&x\ge0\\0&\text{else}\end{cases}
Loss is the Euclidean loss
\dfrac{d\L}
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

    for i in self.Input_layer: self.next[i], self.prev[i] = self.Output_layer, []
    for o in self.Output_layer: self.prev[o], self.next[o] = self.Input_layer, []

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

    delta_b, delta_w = {}, {}

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
      delta_b[c] = -learning_rate * self.bias_gsum[c] / (self.bias_gsqr[c]**(1/2)+1)

      for p in self.prev[c]:
        par_a[p] += par_a[c] * (1-a[c]**2) * self.weight[(p,c)]
        gweight = par_a[c] * (1-a[c]**2) * a[p]
        self.weight_gsum[(p,c)] = \
         (1-dsum)*np.sum(gweight)/batch_size + dsum*self.weight_gsum[(p,c)]
        self.weight_gsqr[(p,c)] = \
         (1-dsqr)*np.sum(gweight**2)/batch_size + dsqr*self.weight_gsqr[(p,c)]
        delta_w[(p,c)] = \
         -learning_rate * self.weight_gsum[(p,c)] / (self.weight_gsqr[(p,c)]**(1/2)+1)

        cnt[p] += 1
        if cnt[p] == len(self.next[p]):
          q.append(p)

    return delta_w, delta_b

  def update(self, X, Y, batch_size, learning_rate, dsum=0.9, dsqr=0.9):
    delta_w, delta_b = \
     self.__backward(X, Y, batch_size, learning_rate, dsum, dsqr)
    for w in self.weight:
        self.weight[w] += delta_w[w]
    for p in self.neurons:
        self.bias[p] += delta_b[p]

  def addLayer(self, mid_size, UP, DOWN):
    Mid_layer = [self.idcnt+m for m in range(mid_size)]
    self.neurons.update([self.idcnt+m for m in range(mid_size)])
    self.idcnt+=mid_size

    for m in Mid_layer:
      self.bias[m] = np.random.uniform(-0.1, 0.1)
      self.bias_gsum[m], self.bias_gsqr[m] = 0, 0

      self.prev[m] = UP
      for u in UP:
        self.weight[(u,m)] = np.random.uniform(-0.1, 0.1)
        self.weight_gsum[(u,m)], self.weight_gsqr[(u,m)] = 0, 0

      self.next[m] = DOWN
      for v in DOWN:
        self.weight[(m,v)] = np.random.uniform(-0.1, 0.1)
        self.weight_gsum[(m,v)], self.weight_gsqr[(m,v)] = 0, 0

    for u in UP:
      self.next[u] = Mid_layer
    for v in DOWN:
      self.prev[v] = Mid_layer

    for u in UP:
      for v in DOWN:
        self.weight.pop((u,v))
        self.weight_gsum.pop((u,v))
        self.weight_gsqr.pop((u,v))
    return Mid_layer

  def train(self, x, y, batch_size, epochs, learning_rate):
    assert len(x) == len(y)
    l = len(x)
    for epoch in range(epochs):
      X, Y = x.copy(), y.copy()
      data=[(X[_], Y[_]) for _ in range(l)]
      random.shuffle(data)
      for _ in range(l):
        X[_],Y[_]=data[_]
      loss = 0
      for batch in range(int(l / batch_size)):
        L, R = batch * batch_size, (batch + 1) * batch_size
        x_train, y_train = X[L:R], Y[L:R]
        self.update(x_train, y_train, batch_size, learning_rate)
        output = self.evaluate(x_train)
        loss += np.sum(((y_train-output) ** 2), axis=(0,1))
      loss = ((loss) ** 0.5) / (int(l / batch_size) * batch_size)
      print(f"Epoch {epoch}/{epochs}, Loss:{loss}")

  def parallel_train(self, x, y, batch_size, epochs=10, learning_rate=0.1):
    assert len(x) == len(y)
    l = len(x)
    for epoch in range(epochs):
      X, Y = x.copy(), y.copy()
      data=[(X[_], Y[_]) for _ in range(l)]
      random.shuffle(data)
      for _ in range(l):
        X[_],Y[_]=data[_]

      k = int(l / batch_size)
      proc = 1
      def train_proc(mod, X_split, Y_split):
        tdelta_w = {w: 0 for w in mod.weight}
        tdelta_b = {q: 0 for q in mod.neurons}
        nonlocal proc, k, batch_size, learning_rate
        for c in range(int(k / proc)):
          delta_w, delta_b = mod.__backward(
            X_split[c*batch_size:(c+1)*batch_size],
            Y_split[c*batch_size:(c+1)*batch_size],
            batch_size, learning_rate, 0.9, 0.9
          )
          for w in self.weight:
            self.weight[w] += delta_w[w]
            tdelta_w[w] += delta_w[w]
          for p in self.neurons:
            self.bias[p] += delta_b[p]
            tdelta_b[p] += delta_b[p]
        return tdelta_w, tdelta_b

      with ThreadPoolExecutor(max_workers=proc) as executor:
        handles = [
          executor.submit(train_proc, self,
            X[b*int(k/proc)*batch_size:(b+1)*int(k/proc)*batch_size],
            Y[b*int(k/proc)*batch_size:(b+1)*int(k/proc)*batch_size])
            for b in range(proc)
          ]

      results = [f.result() for f in handles]

      delta_w = {w: np.mean([res[0][w] for res in results], axis=0) for w in self.weight}
      delta_b = {n: np.mean([res[1][n] for res in results], axis=0) for n in self.neurons}
      for w in self.weight:
        self.weight[w] += delta_w[w]
      for p in self.neurons:
        self.bias[p] += delta_b[p]
      output = self.evaluate(X)
      loss = (np.sum(((Y-output)**2), axis=(0,1)) ** 0.5)\
       / (int(l / batch_size) * batch_size)
      print(f"Epoch {epoch}/{epochs}, Loss:{loss}")

X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0],[1],[1],[0]])
mod=Model(2, 1)
mid1=mod.addLayer(4, mod.Input_layer, mod.Output_layer)
mod.evaluate(X)
mod.parallel_train(X, Y, 4, 500, 0.1)
print(X, Y)
mod.evaluate(X)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import networkx as nx  # Helps with graph layout

class NetworkVisualizer:
    def __init__(self, model):
        self.model = model
    
    def plot_network(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#f0f0f0')

        neurons = list(self.model.neurons)  # Convert neurons to list
        pos = {}

        # Create a NetworkX graph representation
        G = nx.DiGraph()
        for neuron in neurons:
            G.add_node(neuron)

        for (u, v), w in self.model.weight.items():
            G.add_edge(u, v, weight=0.1)

        num_inputs = len(self.model.Input_layer)
        num_outputs = len(self.model.Output_layer)

        for i, n in enumerate(self.model.Input_layer):
            pos[n] = (-1 + 2 * (i / max(1, num_inputs - 1)), -1, 0)  # Fixed y = -1

        for i, n in enumerate(self.model.Output_layer):
            pos[n] = (-1 + 2 * (i / max(1, num_outputs - 1)), 1, 0)  # Fixed y = 1


        # Compute graph layout in 3D (spring layout)
        pos_2d = nx.spring_layout(G, seed=42, dim=3, pos=pos, fixed=pos.keys(), threshold=0.000001)  # 3D spring layout
        for n, p in pos_2d.items():
            pos[n] = p

        # Draw neurons
        for n, p in pos.items():
            ax.scatter(p[0], p[1], p[2], color='gray', s=100, edgecolors='black', linewidth=0.8)
            ax.text(p[0], p[1], p[2], n, color='black', fontsize=8, ha='center')

        # Draw edges with weight-based styling
        for (u, v) in G.edges:
            x_vals = [pos[u][0], pos[v][0]]
            y_vals = [pos[u][1], pos[v][1]]
            z_vals = [pos[u][2], pos[v][2]]
            
            w = abs(self.model.weight[(u, v)])
            alpha = min(0.8, 0.2 + 0.6 * w)
            linewidth = 1 + 3 * w
            ax.plot(x_vals, y_vals, z_vals, color='black', alpha=alpha, linewidth=linewidth)

        # Set labels and visualization tweaks
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Graph-Based Neural Network", fontsize=14, fontweight='bold')

        # Remove axis lines for a cleaner look
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # Rotate animation
        def update(frame):
            ax.view_init(elev=20, azim=frame)

        #ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
        plt.show()
        
vis1 = NetworkVisualizer(mod)
vis1.plot_network()
