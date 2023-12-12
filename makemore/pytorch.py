import torch
import torch.nn.functional as F


# creating torch modules

class Linear:
    def __init__(self, in_features, out_features, bias=True, dtype=None):
        self.gain = torch.randn(
            (in_features, out_features), dtype=dtype)
        self.bias = torch.randn(
            out_features, dtype=dtype) if bias else None

    def __call__(self, input: torch.Tensor):
        out = input @ self.gain
        if self.bias is not None:
            out += self.bias
        return out

    def parameters(self):
        return [self.gain] + ([] if self.bias is None else [self.bias])


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class BatchNorm:
    """
    One-Dimensional Batch Normalization
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # buffers (not trained with backprop)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def parameters(self):
        return [self.gamma, self.beta]

    def __call__(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            x_hat = (x - mean) / (var + self.eps).sqrt()
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * \
                    self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * \
                    self.running_var + self.momentum * var
        else:
            x_hat = (x - self.running_mean) / \
                    (self.running_var + self.eps).sqrt()

        return self.gamma * x_hat + self.beta


if __name__ == "__main__":
    g = torch.Generator().manual_seed(123)
    max_steps = 5000
    emb_size = 50
    hl_size = 100  # hidden layer size
    ctx_len = 5
    vocab_size = 10
    # same optimization as last time
    batch_size = 32
    losses = []
    ud = []
    lre = torch.linspace(-4, 0, max_steps)
    lri = 10 ** lre

    x = torch.randint(0, vocab_size, (1000, ctx_len), generator=g)
    y = torch.randint(0, vocab_size, (1000,), generator=g)
    C = torch.randn((vocab_size, emb_size), dtype=torch.float32, generator=g)

    layers = [
        Linear(emb_size * ctx_len, hl_size,
               bias=False), BatchNorm(hl_size), Tanh(),
        Linear(hl_size, hl_size, bias=False), BatchNorm(hl_size), Tanh(),
        Linear(hl_size, vocab_size, bias=False), BatchNorm(vocab_size),
    ]
    with torch.no_grad():
        emb = C[x]
        emb = C[x].view(emb.shape[0], -1)

        for layer in layers:
            # print(f"{layer.__class__.__name__} layer:  {emb.shape}")
            emb = layer(emb)

    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    # number of parameters in total
    # print(sum(p.nelement() for p in parameters), len(parameters))
    for p in parameters:
        p.requires_grad = True

    for i in range(max_steps):

        # minibatch construct
        ix = torch.randint(0, x.shape[0], (batch_size,), generator=g)
        Xb, Yb = x[ix], y[ix]  # batch X,Y

        # forward pass
        emb = C[Xb]
        emb = emb.view(emb.shape[0], -1)
        for layer in layers:
            emb = layer(emb)

        logits = emb

        loss = F.cross_entropy(input=logits, target=Yb)  # why
        losses.append(loss)
        # print every 100 steps
        if i % 500 == 0:
            print(f"step({i}/{max_steps}): {loss.item()}")

        # backward pass
        # for layer in layers:
            # layer.out.retain_grad()  # AFTER_DEBUG: would take out retain graph
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        with torch.no_grad():
            for p in parameters:
                p -= lri[i] * p.grad

    print(f"final loss: {losses[-1].item()}")
