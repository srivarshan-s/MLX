from jax import random
from jax import grad
from jax import jit


class LinearRegression:
    def __init__(self, random_state=0, loss_fn=None):
        self.loss_fn = None
        self.weights = None
        self.bias = None
        self.grad_weights = None
        self.grad_bias = None
        self.key = random.PRNGKey(random_state)

        if loss_fn is None:

            def loss_fn(pred, y):
                return ((pred - y) ** 2).mean()

        self.loss_fn = loss_fn

    def fit(self, X, y, lr=0.01, num_iter=1_000):
        self.weights = random.uniform(self.key, shape=(X.shape[1], 1))
        self.bias = random.uniform(self.key, shape=(1,))[0]

        def forward(weights, bias, X, y):
            pred = X.dot(weights) + bias
            return self.loss_fn(pred, y)

        self.forward = jit(forward)
        self.grad_weights = jit(grad(self.forward, argnums=0))
        self.grad_bias = jit(grad(self.forward, argnums=1))

        for _ in range(num_iter):
            d_weights = self.grad_weights(self.weights, self.bias, X, y)
            d_bias = self.grad_bias(self.weights, self.bias, X, y)
            self.weights -= d_weights * lr
            self.bias -= d_bias * lr

    def predict(self, X):
        pred = X.dot(self.weights) + self.bias
        return pred
