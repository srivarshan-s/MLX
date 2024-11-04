from jax.numpy import ndarray as JaxArray

def meanAbsoluteError(pred: JaxArray, y: JaxArray):
    return (abs(pred - y)).mean()

def meanSquaredError(pred: JaxArray, y: JaxArray):
    return ((pred - y) ** 2).mean()