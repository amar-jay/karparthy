import torch
from typing import Union
import sys
import numpy as np

class Tensor:
    """
    just a unit tensor
    """
    def __init__(self, data,_prev=set(), _backward=lambda:None):
        self.data = data
        self.grad = 0
        self._prev = _prev if _prev else set()
        self._backward = _backward

    def __add__(self, other:Union['Tensor', int, float]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)

        y = self.data + other.data
        y = Tensor(y, {self, other})

        def _backward():
            self.grad += y.grad
            other.grad += y.grad
            return 
        y._backward = _backward

        return y

    def __mul__(self, other:Union['Tensor', int, float]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)

        y = self.data * other.data
        y= Tensor(y, {self, other})

        def _backward():
            self.grad += other.grad * y.grad
            other.grad += self.grad * y.grad
            return
        y._backward = _backward

        return y

    def __pow__(self, n) -> 'Tensor':
        y = self.data ** n
        y = Tensor(y, {self})

        def _backward():
            self.grad += n * (self.data ** (n-1)) * y.grad
            return
        y._backward = _backward

        return y

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self) -> str:
        data = self.data
        grad = self.grad
        
        return f"Tensor<{data=}, {grad=}>" if self.grad>0 else f"Tensor<{data=}>" 

        
        
    def backward(self):
        children = set()
        children = []
        visited = set()

        def topo(node:Tensor):
            if node not in visited:
                visited.add(node)
                if node._prev is None:
                    print("-", node.data)
                    sys.exit(1)
                for child in node._prev:
                    topo(child)
                children.append(node)

        topo(self)

        self.grad = 1
        for node in children:
            node._backward()
        return

# ----------------- Activation Functions ---------------------------
def tanh(x: Tensor):
    y = np.tanh(x.data)
    y = Tensor(y, {x})

    def _backward():
        x.grad += (1-y.grad**2) * y.grad

    y._backward = _backward

    return y


def sigmoid(x: Tensor):
    y = 1/(1+np.exp(-x.data))
    y = Tensor(y, {x})

    def _backward():
        x.grad += x.data*(1-x.data) * y.grad

    y._backward = _backward

    return y


def relu(x):
    y = Tensor(0 if x.data < 0 else x.data, {x})

    def _backward():
        x.grad += (y.data > 0) * y.grad

    y._backward = _backward
    return y


def forward_test1(a, b):
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + relu(b + a)
    d = d + 3 * d + relu(b - a)
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    return a, b, g

if __name__ == "__main__":
    print("---"*5, "forward pass", "---"*5)

    a = Tensor(21)
    b = Tensor(23)

    c = a + b
    print(f"c=a+b {b=}\t{a=}")

    d = 75
    e = c + d
    print(f"e=c+d {c=}\t{d=}")

    g = e * 0.5
    print(f"g=e * 0.5 \t{e=}")

    f = tanh(g)
    print(f"h=tanh(g) {g=}")

    h = sigmoid(f)
    print(f"h=sigmoid(f) {h=}")

    i = relu(h)
    print(f"i=relu(f) {i=}")

    i.backward()

    print("---"*5, "after backward pass", "---"*5, "\n")

    print(f"{i=}")
    print(f"i=relu(f) {i=}")
    print(f"h=tanh(g) {g=}")
    print(f"g=e * 0.5 \t{e=}")
    print(f"e=c+d {c=}\t{d=}")
    print(f"c=a+b {b=}\t{a=}")

    test_more_ops()
    
