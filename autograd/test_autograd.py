import torch
from torch import tensor

def test_autograd():
    torch_grads = pytorch()
    autograd_grads = this()
    for t, a in zip(torch_grads,autograd_grads):
        assert t - a < 1e-10

def this():
    a = Tensor([100.0])
    
    b = Tensor([200.0])
    c = a + b
    d = Tensor([5.0])
    
    e = c * d
    f = Tensor([0.1])
    g = e / f
    h = g ** 2
    i  = -h
    j = Tensor([0.9])
    k = j-i
    l = log(k)
    m = exp(l)
    n = sigmoid(m ** -0.5)
    o = tanh(n)
    p = relu(o)
    
    params = [p, o,n, m,l,k,j,i, h,g,f,e,d,c,b,a]
    p.zero_grad()
    p.backward()
    return [i.grad for i in params]

def pytorch():
    
    a = tensor([100.0], requires_grad=True)
    b = tensor([200.0], requires_grad=True)
    c = a + b
    d = tensor([5.0], requires_grad=True)
    e = c * d
    f = tensor([.1], requires_grad=True)
    g = e / f
    h = g ** 2
    i  = -h
    j = tensor([0.9], requires_grad=True)
    k = j-i
    l = torch.log(k)
    m = torch.exp(l)
    n = torch.sigmoid(m ** -0.5)
    o = torch.tanh(n)
    p = torch.relu(o)
    params = [p, o,n, m,l,k,j,i, h,g,f,e,d,c,b,a]
    
    for i in params:
        i.retain_grad()
    p.backward()
    return [i.grad.numpy() for i in params]