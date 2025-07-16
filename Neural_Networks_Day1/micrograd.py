import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
            
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

def draw_graph(root):
    plt.figure(figsize=(10, 6))
    
    # Build node hierarchy
    nodes = {}
    def build_nodes(v, depth=0):
        if v not in nodes:
            nodes[v] = {'depth': depth, 'children': []}
            for child in v._prev:
                nodes[v]['children'].append(child)
                build_nodes(child, depth+1)
    build_nodes(root)
    
    # Assign positions
    max_depth = max(n['depth'] for n in nodes.values())
    pos = {}
    for depth in range(max_depth + 1):
        level_nodes = [v for v in nodes if nodes[v]['depth'] == depth]
        for i, node in enumerate(level_nodes):
            pos[node] = (depth, -i)
    
    # Draw edges
    for node in nodes:
        for child in nodes[node]['children']:
            plt.plot([pos[node][0], pos[child][0]], 
                     [pos[node][1], pos[child][1]], 
                     'k-', alpha=0.3)
    
    # Draw nodes
    for node, (x, y) in pos.items():
        plt.gca().add_patch(Rectangle((x-0.4, y-0.2), 0.8, 0.4, 
                           facecolor='white', edgecolor='k'))
        plt.text(x, y, f"{node.label}\ndata: {node.data:.2f}\ngrad: {node.grad:.2f}", 
                ha='center', va='center', fontsize=8)
        if node._op:
            plt.text(x+0.5, y, node._op, color='red', ha='left', va='center')
    
    plt.axis('off')
    plt.title("Computation Graph")
    plt.tight_layout()
    plt.show()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def test_expression():
    print("Testing basic expression...")
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L.backward()
    print("Drawing computation graph...")
    draw_graph(L)

def test_training():
    print("\nTesting neural network training...")
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    model = MLP(3, [4, 4, 1])
    
    print("Initial predictions:")
    for x in xs:
        print(f"Input: {x}, Output: {model(x).data:.4f}")

    for k in range(20):
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        
        for p in model.parameters():
            p.data += -0.1 * p.grad
        
        print(f"Step {k}, Loss: {loss.data:.4f}")

    print("\nFinal predictions:")
    for x, y in zip(xs, ys):
        print(f"Input: {x}, Predicted: {model(x).data:.4f}, Actual: {y}")

if __name__ == "__main__":
    print("Micrograd Demo - Clean Version")
    print("=============================")
    print("This version uses only matplotlib (no Graphviz needed)")
    
    # First clean up any existing Graphviz installations
    try:
        import graphviz
        print("\nWARNING: Graphviz is installed but not needed. Recommend:")
        print("pip uninstall graphviz")
    except ImportError:
        pass
    
    test_expression()
    test_training()
    print("\nDone! All tests completed successfully.")