from typing import Union
import numpy as np

class Tensor:
    def __init__(self, arr=[], _children=set(), _backward=None):
        if not isinstance(self.data, np.ndarray):
            if isinstance(self.data, list):
                arr = np.array(arr)
            else:
                raise ValueError(f'data should be of type "numpy.ndarray" or a scalar,but received {type(arr)}')

        self.data = arr

        self.dtype = self.dtype
        self._children = _children
        self._backward = _backward
        self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape)

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

    def __mul__(self, other:Union['Tensor', int, float]):
        """
            dot and scalar product
        """
        if isinstance(other, (int, float)):
            out = Tensor(other*self.data, (self,))
        else:
            out = Tensor(self.data @ other.data, (self, ))

        def _backward():
            if isinstance(other, (int, float)):
                self.grad += other * out.grad
                return

            if self.data.size == other.data.size:
                self.grad += out.grad @ other.grad.T # works for two dimensional but fails for the rest 
                other.grad += out.grad @ self.grad.T
            else:
                raise NotImplementedError # understanding how matrix multiplcation works

            return
        out._backward = _backward
        return out

    def __pow__(self, _):
        raise NotImplementedError

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

    def dtype(self, _dtype):
        return self.data.astype(_dtype)

    def __repr__(self) -> str:
        data = self.data
        grad = self.grad
        
        return f"Tensor<{data.shape=}, {grad=}>" if self.grad>0 else f"Tensor<{data=}>" 



