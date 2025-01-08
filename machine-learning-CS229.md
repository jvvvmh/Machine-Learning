





## 1 Supervised learning

### Linear regression

#### Least mean square algorithm

$$
\theta:=\theta+\alpha \sum_{i=1}^n\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right) x^{(i)}
$$

This method looks at every example in the entire training set on every step, and is called **batch gradient descent**.


$$
\theta:=\theta+\alpha\left(y^{(i)}-h_i\left(x^{(i)}\right)\right) x^{(i)}
$$

Each time we encounter a training example, we update the parameters according to the gradient of the error with respect to that single training example only. This algorithm is called **stochastic gradient descent** (also incremental gradient descent).

#### Normal equation

$$
\begin{aligned}
\nabla_\theta J(\theta) & =\nabla_\theta \frac{1}{2}(X \theta-\vec{y})^T(X \theta-\vec{y}) \\
& =\frac{1}{2} \nabla_\theta\left((X \theta)^T X \theta-(X \theta)^T \vec{y}-\vec{y}^T(X \theta)+\vec{y}^T \vec{y}\right) \\
& =\frac{1}{2} \nabla_\theta\left(\theta^T\left(X^T X\right) \theta-\vec{y}^T(X \theta)-\vec{y}^T(X \theta)\right) \\
& =\frac{1}{2} \nabla_\theta\left(\theta^T\left(X^T X\right) \theta-2\left(X^T \vec{y}\right)^T \theta\right) \\
& =\frac{1}{2}\left(2 X^T X \theta-2 X^T \vec{y}\right) \\
& =X^T X \theta-X^T \vec{y}
\end{aligned}
$$

In the third step, we used the fact that $a^T b=b^T a$, and in the fifth step used the facts $\nabla_x b^T x=b$ and $\nabla_x x^T A x=2 A x$ for symmetric matrix $A$. To minimize $J$, we set its derivatives to zero, and obtain the normal equations:
$$
X^T X \theta=X^T \vec{y}
$$

Thus, the value of $\theta$ that minimizes $J(\theta)$ is given in closed form by the equation
$$
\theta=\left(X^T X\right)^{-1} X^T \vec{y} \cdot
$$

#### Probabilistic interpretation

 Likelihood function:
$$
L(\theta)=L(\theta ; X, \vec{y})=\vec{p}(\vec{y} \mid X ; \theta)
$$

Note that by the independence assumption on the $\epsilon^{(t)}$ 's (and hence also the $y^{(i)}$ 's given the $x^{(i)}$ 's), this can also be written
$$
\begin{aligned}
L(\theta) & =\prod_{i=1}^n p\left(y^{(i)} \mid x^{(i)} ; \theta\right) \\
& =\prod_{i=1}^n \frac{1}{\sqrt{2 \pi \sigma}} \exp \left(-\frac{\left(y^{(i)}-\theta^T x^{(i)}\right)^2}{2 \sigma^2}\right)
\end{aligned}
$$

$$
=n \log \frac{1}{\sqrt{2 \pi} \sigma}-\frac{1}{\sigma^2} \cdot \frac{1}{2} \sum_{i=1}^n\left(y^{(i)}-\theta^T x^{(i)}\right)^2
$$

Maximizing log-likelihood $\ell(\theta)$ gives the same answer as minimizing
$$
\frac{1}{2} \sum_{i=1}^n\left(y^{(i)}-\theta^T x^{(i)}\right)^2
$$


## 2 Classication and logistic regression

### Logistic regression

$$
\begin{gathered}
h_\theta(x)=g\left(\theta^T x\right)=\frac{1}{1+e^{-\theta^T x}}, \\
g(z)=\frac{1}{1+e^{-z}}
\end{gathered}
$$
$\theta^T x$ is called **logit**.

As before, we are keeping the convention of letting $x_0=1$, so that $\theta^T x=\theta_0+\sum_j \theta_j x_j$
$$
\begin{aligned}
g^{\prime}(z) & =\frac{d}{d z} \frac{1}{1+e^{-z} x} \\
& =\frac{1}{\left(1+e^{-z}\right)^2}\left(e^{-z}\right) \\
& =\frac{1}{\left(1+e^{-z}\right)} \cdot\left(1-\frac{1}{\left(1+e^{-z}\right)}\right) \\
& =g(z)(1-g(z))
\end{aligned}
$$


Let us assume that
$$
\begin{aligned}
& P(y=1 \mid x ; \theta)=h_\theta(x) \\
& P(y=0 \mid x ; \theta)=1-h_\theta(x)
\end{aligned}
$$

$$
\begin{aligned}
L(\theta) & =p(\vec{y} \mid X ; \theta) \\
& =\prod_{i=1}^n p\left(y^{(i)} \mid x^{(i)} ; \theta\right) \\
& =\prod_{i=1}^n\left(h_\theta\left(x^{(i)}\right)\right)^{y^{(i)}}\left(1-h_\theta\left(x^{(i)}\right)\right)^{1-y^{(i)}}
\end{aligned}
$$

Maximize the log likelihood
$$
\ell(\theta)=\log L(\theta)=\sum_{i=1}^n y^{(i)} \log h\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h\left(x^{(i)}\right)\right)
$$
Stochastic gradient descent rule
$$
\theta_j:=\theta_j+\alpha\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right) x_j^{(i)}
$$

### Perceptron learning algorithm

Consider modifying the logistic regression method to "force" it to output values that are either 0 or 1 or exactly. To do so, it seems natural to change the definition of $g$ to be the threshold function:
$$
g(z)= \begin{cases}1 & \text { if } z \geq 0 \\ 0 & \text { if } z<0\end{cases}
$$

If we then let $h_\theta(x)=g\left(\theta^T x\right)$ as before but using this modified definition of $g$, and if we use the update rule
$$
\theta_j:=\theta_j+\alpha\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right) x_j^{(i)}
$$
then we have the **perceptron learning algorithn**.
In the 1960s, this "perceptron" was argued to be a rough model for how individual neurons in the brain work. Note however that even though the perceptron may be cosmetically similar to the other algorithms we talked about, it is actually a very different type of algorithm than logistic regression and least squares linear regression; in particular, it is difficult to endow the perceptron's predictions with meaningful probabilistic interpretations, or derive the perceptron as a maximum likelihood estimation algorithm.



