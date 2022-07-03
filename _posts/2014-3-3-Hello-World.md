---
layout: post
title: You're up and running!
---

Next you can update your site name, avatar and other options using the _config.yml file in the root of your repository (shown below).

![_config.yml]({{ site.baseurl }}/images/config.png)

The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.

# Simple MLP Forward Backward Equations

- [Simple MLP Forward Backward Equations](#simple-mlp-forward-backward-equations)
  - [Background: Matrix Differentiation](#background-matrix-differentiation)
    - [Linear Case](#linear-case)
    - [Non-Linear Case](#non-linear-case)
  - [Forward Pass](#forward-pass)
  - [Backward Pass](#backward-pass)
    - [Output Layer](#output-layer)
    - [Hidden Layer](#hidden-layer)
    - [Input Layer](#input-layer)
    - [Self-attention Layer](#self-attention-layer)
  - [References](#references)
  - [Appendix](#appendix)

## Background: Matrix Differentiation  

### Linear Case

For matrix $A \in \Reals^{m \times n}$ and $B \in \Reals^{n \times k}$, define $C=AB \in \Reals^{m \times k}$. If we know the gradient of output $C$ as $\frac{\partial L}{\partial C} \in \Reals^{k \times m}$. The gradient with respect to $B$ is 
$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C} A$$ (0.1)
The gradient with respect to $A$ is 
$$\frac{\partial L}{\partial A} = B \frac{\partial L}{\partial C} $$ (0.2)

Proof of Eq. (0.1). 
$$\begin{split}\frac{\partial C_{ij}}{\partial B_{mn}} &= \frac{\partial (\sum_k A_{ik}B_{kj})}{\partial B_{mn}} \\ 
&= \sum_k A_{ik} \delta_{km} \delta_{jn} \\
&= A_{im} \delta_{jn}  
\end{split}$$ (0.3)
$$\begin{split}\frac{\partial L}{\partial B_{mn}} &= \sum_{ij} \frac{\partial L}{\partial C_{ij}}\frac{\partial C_{ij}}{\partial B_{mn}} \\
&= \sum_{ij} \frac{\partial L}{\partial C_{ij}}  A_{im} \delta_{jn} \\
&= \sum_{i} \frac{\partial L}{\partial C_{in}}  A_{im} 
\end{split}$$ (0.4)
So Eq (0.4) is equivalent to 
$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C} A$$
q.e.d.

Proof of Eq. (0.2)
$$C^T = B^TA^T$$ (0.5)
apply Eq. (0.1) on Eq. (0.5)
$$ \frac{\partial L}{\partial A^T} = \frac{\partial L}{\partial C^T} B^T$$ (0.6)
Transpose both sides
$$ \frac{\partial L}{\partial A} =  B\frac{\partial L}{\partial C}$$ (0.7)
q.e.d
Or check appendix for alternative proof.

### Non-Linear Case
For matrix $A \in \Reals^{m \times n}$ and $B \in \Reals^{n \times k}$, define $C=\sigma(AB) \in \Reals^{m \times k}$, where $\sigma(x)$ is a element-wise non-linear function. If we know the gradient of output $C$ as $\frac{\partial L}{\partial C} \in \Reals^{k \times m}$. The gradient with respect to $B$ is 
$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\cdot {\sigma^{'}}^{T}  A$$ (0.8)
The gradient with respect to $A$ is 
$$\frac{\partial L}{\partial A} = B \frac{\partial L}{\partial C}\cdot {\sigma^{'}}^{T}  $$ (0.9)

Proof of Eq. (0.8). 
$$\begin{split}\frac{\partial C_{ij}}{\partial B_{mn}} &= \frac{\partial \sigma(\sum_k A_{ik}B_{kj})}{\partial B_{mn}} \\ 
&= \sigma^{'}(\sum_k A_{ik}B_{kj}) \frac{\partial \sigma (\sum_k A_{ik}B_{kj})}{\partial B_{mn}} \\ 
&= \sigma^{'}(\sum_k A_{ik}B_{kj}) \sum_k A_{ik} \delta_{km} \delta_{jn} \\
&= \sigma^{'}_{ij}A_{im} \delta_{jn}  
\end{split}$$ (0.10)
$$\begin{split}\frac{\partial L}{\partial B_{mn}} &= \sum_{ij} \frac{\partial L}{\partial C_{ij}}\frac{\partial C_{ij}}{\partial B_{mn}} \\
&= \sum_{ij} \frac{\partial L}{\partial C_{ij}} \sigma^{'}_{ij} A_{im} \delta_{jn} \\
&= \sum_{i} \frac{\partial L}{\partial C_{in}} \sigma^{'}_{in} A_{im} 
\end{split}$$ (0.11)
So Eq (0.11) is equivalent to 
$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\cdot {\sigma^{'}}^{T} A$$
where $\cdot$ is element-wise multiplication.

q.e.d.

Proof of Eq. (0.7) is similar to proof of Eq. (0.6)

## Forward Pass 

Let's assume we have an 2-layer MLP neural network. 

Input is a column vector $x_0 \in \Reals^{v \times 1}$, where $v$ is a finite dimension. E.g. vocabulary dimension.

The input layer converts the finite dimension to infinite $d$ by matrix $W_1 \in \Reals^{d \times v}$.

Assume all the point-wise non-linear activation functions are $\sigma(x) \in \Reals^{d \times 1}$. 

The output $x_1$ is defined as 
$$x_1 = \sigma(W_1 x_0)$$ (1)

Similarly, the inter-layer weight $W_2 \in \Reals^{d \times d}$ transform the input $x_1$ to the output $x_2$ by 
$$x_2 = \sigma(W_2 x_1)$$ (2)

Lastly, the output layer matrix $W_3 \in \Reals^{,v \times d}$ convert the infinite dimension $d$ into finite $v$.

$$x_3 = \sigma(W_3 x_2)$$ (3)

Add the MSE loss layer as 
$$ L = \Vert x_3 - y \Vert ^2 $$ (4)

## Backward Pass 

Starting from the loss $L$, we calculate $\frac{\partial L}{\partial x_3} \in \Reals^{1 \times v}$ as:
$$ \frac{\partial L}{\partial x_3} = 2 (x_3 - y)^T $$ (5)

Following the 3 desiderata as in stated by Yang et al [1], i.e.
1. Every (pre)activation vector in a network should have $\Theta(1)$-sized coordinates.
2. Neural network output should be $O(1)$.
3. All parameters should be updated as much as possible (in terms of scaling in width) without
leading to divergence.

Since $x_0, x_3 \in \Reals^{v \times 1}$ and $x_1, x_2 \in \Reals^{d \times 1}$ are all $\Theta(1)$-sized coordinates, we can see $\frac{\partial L}{\partial x_3}$ is of size $\Theta(1)$.

Since we know from Yang et al [1] that the output layer should have $W_3$ of size $\Theta(1/d)$ to make $x_3$ of size $\Theta(1)$. This is because $x_2$ and $W_3$ are correlated and $x_2$ are of size $\Theta(1)$. Using law of large number, we can see $W_3 x_2$ is of coordinate size $\Theta(1)$. 

### Output Layer

Calculate the $\frac{\partial L}{\partial W_3}$ according to Eq. (0.9):
$$
\begin{split}
\frac{\partial L}{\partial W_3} &= x_2 \frac{\partial L}{\partial x_3}\cdot {\sigma^{'}}^{T}(W_3 x_2) \\ 
&= 2 x_2 (x_3 - y)^T \cdot {\sigma^{'}}^{T}(W_3 x_2)
\end{split} $$ (6)

where $\cdot$ is point-wise multiplication. We define vector $c_2=2 (x_3 - y) \cdot \sigma^{'}(W_3 x_2) \in \Reals^{v \times 1}$. It is easy to see $c_2$ has coordinate size of $\Theta(1)$.  And $\frac{\partial L}{\partial W_3}$ is simplified to 
$$ \frac{\partial L}{\partial W_3} = x_2 c_2^T $$ (7)

According to the 3 desiderata, we want 
$(W_3 - \eta \Delta W_3)x_2$ of coordinate size $\Theta(1)$, which implies both $W_3 x_2$ and $\eta \Delta W_3 x_2$ have coordinate size of $\Theta(1)$. So, $W_3$ has to be initialized with coordinate size of $\Theta(1/d)$. While for $ \eta \Delta W_3 x_2 = \eta c_2x_2^T x_2$. For SGD, we need to set $\eta=1/d$, so $x_2^T x_2 / d$ has coordinate size of $\Theta(1)$. Note, $x_2^T$ is the $x_2$ in the previous step. For Adam, $x_2^T x_2$ needs a constant $1/d$ to be independent of $d$. 


Calculate $\frac{\partial L}{\partial x_2} \in \Reals^{1 \times d}$ according to Eq. (0.8)
$$\begin{split}\frac{\partial L}{\partial x_2}  &=  \frac{\partial L}{\partial x_3}\cdot {\sigma^{'}}^{T}(W_3 x_2) W_3 \\ &= c_2^T W_3\end{split}$$  (8)

Since $v$ is finite, $W_3$ is of order $\Theta(1/d)$, $\frac{\partial L}{\partial x_2}$ is of order $\Theta(1/d)$ 

### Hidden Layer

Calculate the $\frac{\partial L}{\partial W_2}$ according to Eq. (0.9):
$$
\begin{split}
\frac{\partial L}{\partial W_2} &= x_1 \frac{\partial L}{\partial x_2}\cdot {\sigma^{'}}^{T}(W_2 x_1) \\
&= x_1 (c^T_2 W_3) \cdot {\sigma^{'}}^{T} (W_2 x_1)
\end{split} $$ (9)

We define vector $c_1= (c^T_2 W_3)^T \cdot \sigma^{'}(W_2 x_1) \in \Reals^{d \times 1}$ which has coordinate size of $\Theta(1/d)$. 

According to the 3 desiderata, we want 
$(W_2 - \eta \Delta W_2)x_1$ of coordinate size $\Theta(1)$, which implies both $W_2 x_1$ and $\eta \Delta W_2 x_1$ have coordinate size of $\Theta(1)$. So, $W_2$ has to be initialized with coordinate size of $\Theta(1/\sqrt{d})$. While for $ \eta \Delta W_2 x_1 = \eta c_1x_1^T x_1$. For SGD, we need to set $\eta=1$, so $c_1 x_1^T x_1$ has coordinate size of $\Theta(1)$ because $c_1$ has order of $\Theta(1/d)$. Note, $x_1^T$ is the $x_1$ in the previous step. For Adam, $x_1^T x_1$ needs a constant $1/d$ because of the $\Theta(1/d)$ term $c_1$ in the gradient is canceled out by normalization.

Calculate the $\frac{\partial L}{\partial x_1}$ according to Eq. (0.8):
$$\begin{split}\frac{\partial L}{\partial x_1} &= \frac{\partial L}{\partial x_2}\cdot {\sigma^{'}}^{T} (W_2 x_1) W_2 \\ 
&= c_1^T W_2\end{split}$$  (10)

### Input Layer

Calculate the $\frac{\partial L}{\partial W_1}$ according to Eq. (0.9):
$$
\begin{split}
\frac{\partial L}{\partial W_1} &= x_0 \frac{\partial L}{\partial x_1}\cdot {\sigma^{'}}^{T}(W_1 x_0) \\
&= x_0 c_1^T W_2 \cdot {\sigma^{'}}^{T}(W_1 x_0) 
\end{split} $$ (11)

According to the 3 desiderata, we want 
$(W_1 - \eta \Delta W_1)x_0$ of coordinate size $\Theta(1)$, which implies both $W_1 x_0$ and $\eta \Delta W_1 x_0$ have coordinate size of $\Theta(1)$. So, $W_1$ has to be initialized with coordinate size of $\Theta(1)$ because of the finite $x_0$ dimension. While for $ \eta \Delta W_1 x_0 = \eta W_2^T c_1 \cdot \sigma^{'}(W_1 x_0) x_0^T x_0$. $x_0^T x_0$ is of order $\Theta(1)$ because of the finite $v$ dimension. To counter for the extra $\Theta(1/d)$ from the $c_1$ term, for SGD, we need to set $\eta=d$. Note, $x_0^T$ is the $x_0$ in the previous step. For Adam, it needs a constant $1$ because the $\Theta(1/d)$ term $c_1$ in the gradient is canceled out by normalization.

To see why ${x_2^{t-1}}^T$ at $t-1$ correlates with $x_2^t$ at $t$. Calculate the $x_2^t$ by the forward pass Eq. 2 and note that weight matrix $W_2^t = W_2^{t-1} - \eta c_2^{t-1} {x_2^{t-1}}^T$
$$
\begin{split}
x_2^t &= \sigma(W_2^{t} x_1^t)) \\
&= (\sigma((W_2^{t-1} - \eta c_2 {x_2^{t-1}}^T) x_1^t))
\end{split} $$
where we can easily identify that $x_2^t$ contains the term $x_2^{t-1}$, so it has positive corelation with $x_2^{t-1}$.

### Self-attention Layer

From above example, we can see the layer output gradient has coordinate size of $\Theta(1)$ because the $c_1$ term passed down from the output layer in the backward pass. So we have $\frac{\partial L}{\partial O}$ has size $\Theta(1)$

Ignoring the multiple heads, assume the input $x \in \Reals^{s \times d}$ where $s$ is the finite sequence dimension, the matrices $K,Q,V \in \Reals^{d \times d}$. The self-attention layer is
$$O = \sigma(\frac{xQK^Tx^T}{d})xV$$

Note here we use $d$ to scale the attention score as shown later it is necessary.

Calculate the $\frac{\partial L}{\partial \sigma}$ according to Eq. (0.2):
$$
\begin{split}
\frac{\partial L}{\partial \sigma} &= xV \frac{\partial L}{\partial O} 
\end{split} $$ (12)

Calculate the $\frac{\partial L}{\partial xQ}$ according to Eq. (0.9):
$$
\begin{split}
\frac{\partial L}{\partial xQ} &= \frac{K^Tx^T}{d} \frac{\partial L}{\partial \sigma}\cdot {\sigma^{'}}^T
\end{split} $$ (13)

Calculate the $\frac{\partial L}{\partial Q}$ according to Eq. (0.1) and substitute Eq. 12 and Eq 13:
$$
\begin{split}
\frac{\partial L}{\partial Q} &= \frac{\partial L}{\partial xQ} x \\
&= \frac{K^Tx^T}{d} \frac{\partial L}{\partial \sigma}\cdot {\sigma^{'}}^T x \\
&= \frac{K^Tx^T}{d} xV \frac{\partial L}{\partial O} \cdot {\sigma^{'}}^T x \\
\end{split} $$ (14)

According to the 3 desiderata, we want 
$x(Q - \eta \Delta Q)$ of coordinate size $\Theta(1)$, which implies both $xQ$ and $\eta x \Delta Q$ have coordinate size of $\Theta(1)$. So, $Q$ has to be initialized with coordinate size of $\Theta(1/\sqrt{d})$. While for $ \eta x \Delta Q = \eta x x^T {\sigma^{'}} \cdot \frac{\partial L}{\partial O^T} \frac{V^T x^T x K}{d} $. $V^Tx^T x K$ is of order $\Theta(1)$ because of the finite $s$ dimension. $\frac{xx^T}{d}$ is of order $\Theta(1)$ by law of large numbers while the $\frac{\partial L}{\partial O}$ term has order of $\Theta(1/d)$ as we discussed before. So this is the same case as the input layer. For SGD, we need to set $\eta=1$, because $\frac{\partial L}{\partial O}$ term has order of $\Theta(1/d)$. Note, $x_1^T$ is the $x_1$ in the previous step. For Adam, $\eta$ is set to a constant $1/d$ because of the $\Theta(1/d)$ term in the gradient is canceled out by normalization.  

For $\frac{\partial L}{\partial K}$ and $\frac{\partial L}{\partial V}$ cases, they are similar to the $\frac{\partial L}{\partial Q}$ case and we can use the input layer parameterization rules.

## References

1. Yang, Greg, et al. "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." arXiv preprint arXiv:2203.03466 (2022).

## Appendix 

Alternative proof of Eq. (0.2)
$$\begin{split}\frac{\partial C_{ij}}{\partial A_{mn}} &= \frac{\partial (\sum_k A_{ik}B_{kj})}{\partial A_{mn}} \\ 
&= \sum_k B_{kj} \delta_{im} \delta_{kn} \\
&= B_{nj} \delta_{im}  
\end{split}$$ (0.3)
$$\begin{split}\frac{\partial L}{\partial A_{mn}} &= \sum_{ij} \frac{\partial L}{\partial C_{ij}}\frac{\partial C_{ij}}{\partial A_{mn}} \\
&= \sum_{ij} \frac{\partial L}{\partial C_{ij}}  B_{nj} \delta_{im} \\
&= \sum_{j} \frac{\partial L}{\partial C_{mj}}  B_{nj} 
\end{split}$$ (0.4)
So Eq (0.4) is equivalent to 
$$\frac{\partial L}{\partial A} = B \frac{\partial L}{\partial C}$$
q.e.d.
