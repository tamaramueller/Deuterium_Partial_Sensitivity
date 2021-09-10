from collections import defaultdict
import symengine as sy
import numpy as np


class Variable:

    __slots__ = (
        "data",
        "parents",
        "local_gradients",
        "grad",
    )

    def __init__(self, data, parents=(), local_gradients=()):
        self.data = data
        self.parents = frozenset(parents)
        self.local_gradients = local_gradients
        self.grad = 0.0

    def __add__(self, other):
        other = Variable(other) if not isinstance(other, Variable) else other
        data = self.data + other.data
        local_gradients = {self: 1.0, other: 1.0}
        return Variable(
            data=data, parents=(self, other), local_gradients=local_gradients
        )

    def __radd__(self, other):
        other = Variable(other) if not isinstance(other, Variable) else other
        return other.__add__(self)

    def __mul__(self, other):
        if other is self:
            return self.__pow__(2)
        other = Variable(other) if not isinstance(other, Variable) else other
        data = self.data * other.data
        local_gradients = {self: other.data, other: self.data}
        return Variable(
            data=data, parents=(self, other), local_gradients=local_gradients
        )

    def __rmul__(self, other):
        other = Variable(other) if not isinstance(other, Variable) else other
        return other.__mul__(self)

    def __neg__(self):
        data = -self.data
        local_gradients = {self: -1.0}
        return Variable(data=data, parents=(self,), local_gradients=local_gradients)

    def __sub__(self, other):
        other = Variable(other) if not isinstance(other, Variable) else other
        return self.__add__(-other)

    def __rsub__(self, other):
        other = Variable(other) if not isinstance(other, Variable) else other
        return other.__add__(-self)

    def __pow__(self, other):
        other = Variable(other) if not isinstance(other, Variable) else other
        data = self.data ** other.data
        local_gradients = {
            self: other.data * (self.data ** (other.data - 1.0)),
            other: sy.log(self.data) * (self.data ** other.data),
        }
        return Variable(
            data=data, parents=(self, other), local_gradients=local_gradients,
        )

    def __rpow__(self, other):
        other = Variable(other) if not isinstance(other, Variable) else other
        return other.__pow__(self)

    def __truediv__(self, other):
        other = Variable(other) if not isinstance(other, Variable) else other
        return self.__mul__(other.__pow__(-1.0))

    def __rtruediv__(self, other):
        other = Variable(other) if not isinstance(other, Variable) else other
        return other.__mul__(self.__pow__(-1.0))

    def exp(self):
        data = sy.exp(self.data)
        local_gradients = {self: sy.exp(self.data)}
        return Variable(data=data, parents=(self,), local_gradients=local_gradients)

    def log(self):
        data = sy.log(self.data)
        local_gradients = {self: self.data ** (-1.0)}
        return Variable(data=data, parents=(self,), local_gradients=local_gradients)

    def sqrt(self):
        return self.__pow__(0.5)

    def backward(self):
        self.grad = 1
        for node in self._toposort():
            for parent in node.parents:
                local_grad = node.local_gradients[parent]
                parent.grad += local_grad * node.grad

    def _toposort(self):
        postorder = []
        visited = set()

        def dfs(node):
            if node.parents and node not in visited:
                visited.add(node)
                for parent in node.parents:
                    dfs(parent)
                postorder.append(node)

        dfs(self)
        return reversed(postorder)

    def __repr__(self):
        return str(self.data)


def _relu(var):
    var = Variable(var) if not isinstance(var, Variable) else var
    data = sy.Piecewise((0, var.data <= 0), (var.data, var.data > 0))
    local_gradients = {var: sy.Piecewise((0, var.data <= 0), (1, var.data > 0))}
    return Variable(data=data, parents=(var,), local_gradients=local_gradients)


relu = np.vectorize(lambda x: _relu(x))


def get_gradients(variable, wrt=None):
    """Calculate gradients of `variable` with respect to `wrt`.
    This method should generally not be used as it is slow and
    inefficient.
    """
    gradients = defaultdict(lambda: 0.0)

    def compute_gradients(variable, path_value):
        if variable.local_gradients:
            for parent, grad in variable.local_gradients.items():
                to_par = path_value * grad
                if wrt is None or str(parent) in wrt:
                    gradients[str(parent)] += to_par
                compute_gradients(parent, to_par)

    compute_gradients(variable, path_value=1)
    return dict(gradients)
