# Autograd

Inspired with the style [karparthy](https://github.com/karpathy) implemented micrograd. This is an attempt to reverse engineer pytorch autograd on numpy ndarray.

Autograd is a efficient and accurately method of deriving the derivatives of numeric functions by primarily thorugh [chain rule](https://en.wikipedia.org/wiki/Chain_rule_%28probability%29#:~:text=Chain%20rule%20%28probability%29%20-%20Wikipedia%20Chain%20rule%20%28probability%29,distribution%20of%20random%20variables%20respectively%2C%20using%20conditional%20probabilities.) 
and _algorithmic differentiation_. Every operation performed on tensors can be shown as a DAG (directed acylic graph).
Thinking in terms of the DAG, using the chain rule the derivative can only be calculated only if the parent node is found.
We can create a toplogical order similar to that [micrograd](https://github.com/karpathy/micrograd) but taking broadcasting and otehr tensor operations into account.


### References 
- [Tensor product](https://www.youtube.com/watch?v=qp_zg_TD0qE)
- [micrograd](https://github.com/karpathy/micrograd) 
