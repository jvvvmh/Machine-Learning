





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