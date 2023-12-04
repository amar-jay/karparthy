## **Bigram Language Model: A Practical Approach** | [Paper](./assets/rumelhart86.pdf) | [Code](https://github.com/amar-jay/karparthy/blob/main/makemore/bigram.ipynb)

The bigram language model, as implemented in the provided notebook, employs a statistical approach dating back to the early 19th century. It predicts the succeeding character in a sequence based on the preceding character, operating in a two-dimensional space and generating a tensor of character probabilities following one another.

#### **Basic Implementation** 
Initially, the model involves creating a tensor that records the likelihood of pairs of characters occurring. This probability tensor is utilized to determine a sequence of characters, and the overall probability of the entire sequence occurring. The widely adopted metric for evaluating the performance of a bigram model is the Negative Mean Log Likelihood ($NMLL$). A lower $NMLL$ corresponds to lower loss, signifying a higher likelihood of the sequence occurring. However, a limitation arises as the model becomes less manageable with an increasing range of possibilities (pairs of characters), resulting in a frame that grows exponentially, making it challenging to monitor. \
$$ Model\ size = v^c - 1$$ \
$v$ = vocabulary size \
c = context length 

#### **Efficient Enhancement: Tensors and Gradient Descent** | [Paper](./assets/rumelhart86.pdf) 
To address the scalability issues, a more efficient system was devised. This involves expressing the model in the form of an equation: $y=x \times M + c$, where $M$ represents the weights, and $c$ denotes the bias. The process begins with the multiplication of random weights $(M)$ by a one-hot encoding of all characters in the training set $(x)$. Each character is represented by a vector where only its corresponding index is set to 1, and the rest are 0. Subsequently, an activation function $(softmax)$ is applied to the results. The $softmax$ function normalizes the output, converting it into a probability distribution. \
$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$
#### **Gradient Descent**
Following the multiplication and activation, gradient descent is employed with a reasonable learning rate to iteratively adjust the weights. This iterative optimization process minimizes the loss, enhancing the model's ability to predict sequences accurately. Gradient descent computes the gradients of the loss with respect to the weights, indicating the direction in which the weights should be adjusted to reduce the loss.

#### **Addressing Dimensionality Challenges** 
One inherent challenge lies in determining the appropriate dimension. The curse of dimensionality complicates this aspect, as a word sequence for model testing is likely to differ significantly from all word sequences encountered during training. Traditional yet successful approaches, rooted in n-grams (multi-dimensional bigrams), achieve generalization by concatenating very short overlapping sequences observed in the training set.

_It's worth noting that Yoshua Bengio and his colleagues addressed the dimensionality problem in their paper "A Neural Probabilistic Language Model."_

This efficient and scalable approach not only overcomes the challenge of a large number of inputs but also navigates the intricacies of dimensionality, providing a robust foundation for language modeling tasks. It did so by the introduction of **Multi-Layer Perceptron**(MLP)

## Multi-Layer Perceptron | [Paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) | [Code](https://github.com/amar-jay/karparthy/blob/main/makemore/mlp.ipynb)
The problem of multidimensionality that had been plagueing statistical modelling in the 19th centery. Varied ways of resolving it were made. However none was as significant as Bengio and et al. paper **"A Neural Probabilistic Language Model"** which introduced the concept which is known today as MultiLayer Neural Networks. In this paper, it addressed the issue of mulitdimensionality by the implementation of multiple layers. Within each layer, 
