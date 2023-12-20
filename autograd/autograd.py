from typing import Union
import numpy as np

class Tensor:
    def __init__(self, arr=[], _children=set(), _backward=lambda:None):
        if not isinstance(arr, np.ndarray):
            if isinstance(arr, list):
                arr = np.array(arr)
            else:
                raise ValueError(f'data should be of type "numpy.ndarray" or a scalar,but received {type(arr)}')

        self.data = arr

        self.dtype = self.dtype
        self._children = _children
        self._backward = _backward
        self.grad = np.zeros_like(self.data, dtype=np.float64)  # is this really the best way to implement this?

    def zero_grad(self):
        self.grad = np.ones_like(self.data, dtype=np.float64)

    def __add__(self, other:'Tensor'):
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            if self.data.size == other.data.size:
                self.grad += out.grad
                other.grad += out.grad
            else:
                for i, j in zip(self.data.shape, out.data.shape):
                    if i != j:
                        raise ValueError(f"Shapes are different self:{self.data.shape} other:{self.data.shape}")

                if len(self.data.shape) < len(out.data.shape):
                    self.grad += np.sum(out.data, axis=0)
                    other.grad += out.data

        out._backward = _backward

        return out

    def __mul__(self, other:Union['Tensor', int, float]) -> 'Tensor':
        """
            dot and scalar product
        """
        if isinstance(other, (int, float)):
            y = Tensor(other*self.data, (self,other))
            print("from here")
        else:
            if self.data.shape != other.data.shape:
                raise ValueError(f"Shapes are different self:{self.data.shape} other:{other.data.shape}")
            y = Tensor(self.data * other.data, (self, other))

        def _backward():
            if isinstance(other, (int, float)):
                self.grad += other * y.grad
                return

            if self.data.shape == other.data.shape:
                self.grad += other.data * y.grad # works for two dimensional but fails for the rest 
                other.grad += self.data * y.grad
            else:
                raise NotImplementedError # understanding how matrix multiplcation works

            return
        y._backward = _backward
        return y

    def __matmul__(self, other: 'Tensor'):
        if not isinstance(other, Tensor):
            raise ValueError(f'data should be of type "Tensor"  {type(other)}')
        if self.data.shape != other.data.shape:
            raise ValueError(f"Shapes are different self:{self.data.shape} other:{other.data.shape}")
        y = Tensor(self.data @ other.data, (self, ))
        
        def _backward():
            self.grad += np.dot(y.grad, other.grad.T)
            other.grad += np.dot(self.grad.T, y.grad)
        y._backward = _backward
        return y

    def __pow__(self, n):
        if n < 0: # numpy does not support negative exponents 
            y = Tensor(1/(self.data ** -n), (self,))
        else:
            y = Tensor(self.data ** n, (self,))

        def _backward():
            if n-1 < 0: # numpy does not support negative exponents 
                self.grad += ((n / self.data ** -(n-1))) * y.grad
            else:
                self.grad += (n * self.data ** (n-1)) * y.grad
        y._backward = _backward
        return y

    def __div__(self, other:Union['Tensor', int, float]): # other / self
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other * self**-1

    def T(self):
        y = Tensor(self.data.T, (self,))
        def _backward():
            self.grad += y.grad.T

        y._backward = _backward
        return y

    def backward(self):
        children = []
        visited = set()

        def build_topo(node):
            if node not in visited and node is not None:
                visited.add(node)
                if node._children:
                    for child in node._children:
                        build_topo(child)
                children.append(node)
        build_topo(self)
        print(children)

        for child in children:
            if child is None:
                print("null child ", child.data)
                continue
            child._backward()
        return

    def __neg__(self):
        return self * -1

    def __sub__(self, other:'Tensor'):
        return self + (-other)
    
    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other:'Tensor'): # self / other
        return self * other**-1

    def __rtruediv__(self, other:'Tensor'): # other / self
        return other * self**-1

    def dtype(self, _dtype):
        return self.data.astype(_dtype)

    def __repr__(self) -> str:
        data = self.data
        grad = self.grad
        
        return f"Tensor<{data.tolist()}, {grad=}>" if self.grad>0 else f"Tensor<{data.tolist()}>" 


# ---------------------------------- Activation functions --------------------------------
def exp(x:Tensor):
    y = np.exp(x.data)
    y = Tensor(y, (x,))

    def _backward():
        dy = np.exp(x.data)
        x.grad += dy * y.grad
        return

    y._backward = _backward
    return y

def log(x:Tensor):
    y = np.log(x.data)
    y = Tensor(y, (x,))
    def _backward():
        dy = x.data ** -1
        x.grad += dy * y.grad
        return

    y._backward = _backward
    return y

def relu(x:Tensor):
    y = np.maximum(x.data, 0)
    y = Tensor(y, (x,))

    def _backward():
        x.grad[x.data>0] += y.grad[x.data>0]
        return

    y._backward = _backward
    return y


# check if both implementation are equal
def sigmoid(x:Tensor):
    return (Tensor([1])+exp(-x)) ** -1

def sigmoid_2(x:Tensor):
    y = 1/(1+np.exp(-x.data))
    y = Tensor(y, (x,))

    def _backward():
        dy = x.data*(1-x.data)
        x.grad += dy * y.grad
        return 

    y._backward = _backward
    return y

def tanh(x:Tensor):
    return (exp(x) - exp(-x))/(exp(x) + exp(-x))

def tanh_2(x:Tensor):
    y = np.tanh(x.data)
    y = Tensor(y, (x,))

    def _backward():
        dy = (1-y.data**2)
        x.grad += dy * y.grad
        return

    y._backward = _backward
    return y

