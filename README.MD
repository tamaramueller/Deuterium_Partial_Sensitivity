# README

## Overview
Deuterium is an operator overloading-based hybrid automatic differentiation system. This document provides a brief overview of the system components and instructions on how to install the package. An executable version of this documentation is also provided [here](Section_4_System_Overview.ipynb). 

## For paper readers
The jupyter notebooks in the root directory correspond to the sections of the manuscript. HTML versions of the notebooks are also provided, which can be opened without the requirement to install the runtime or wait for calculations to complete.

## Environment Setup
The package has been tested under Ubuntu Linux 20.04 LTS and under MacOS 11.3.1. We recommend using the `conda` package manager to create an isolated environment for the package using the command:

`conda env create -f environment.yml`

## Running Tests
To test whether everything has installed correctly, run the command `pytest .` from the main directory after activating the `conda` environment.

## Basic Usage

Deuterium allows differentiation of functions with respect to symbols and mixing symbols and numbers in expressions.

### Purely symbolic autodiff
In this usage mode, the system calculates closed-form expressions for variables which are represented as symbols and returns an expression representing the derivative with respect to the requested variable. Example:

```py
from deuterium import Variable
import symengine as se
import numpy as np

a, b = se.symbols("a, b")
a = Variable(a)
b = Variable(b)
f = lambda a, b: a ** 2 + 2 * np.log(a * np.sqrt(a * b ** 3)) - np.exp(a + 2 * b) + 5
result = f(a, b)
print(result)

>> 5 + a**2 - exp(a + 2*b) + 2*log(a*(a*b**3)**0.5)

result.backward()

print(a.grad)

>> 2.0*(a*b**3)**0.5*(a*(a*b**3)**0.5)**(-1.0) + 1.0*a*b**3*(a*b**3)**(-0.5)*(a*(a*b**3)**0.5)**(-1.0) + 2.0*a**1.0 - 1.0*exp(a + 2*b)

print(b.grad)

>> 3.0*a**2*b**2.0*(a*b**3)**(-0.5)*(a*(a*b**3)**0.5)**(-1.0) - 2.0*exp(a + 2*b)
```

### Numeric autodiff
If no symbols are used, derivatives can be calculated with respect to numerical quantities. Example:

```py
from deuterium import Variable
import numpy as np

a, b = 2.0, 3.0
a = Variable(a)
b = Variable(b)
f = (
    lambda a, b: a ** 2
    + 2 * np.log(a * np.sqrt(a * b ** 3))
    - np.exp(a / 30.0 + 2 * np.sqrt(b))
    + 5
)
result = f(a, b)
print(result)

>> -19.7749161040171

result.backward()

print(a.grad)

>> 4.36166018294329

print(b.grad)

>> -18.7166239942088
```

### Array computations
The fundamental unit of computation in Deuterium is the `Variable`, which represents a scalar. However, array computations can easily be performed using `numpy`. Deuterium provides utility functions to create arrays of named symbols and to convert `numpy` arrays to Deuterium `Variable` types. Example:

```py
from deuterium import Variable, random_symbols, to_vec
import numpy as np


symbols = random_symbols(n=10, prefix="x")  # generate symbols (x_0,...,x_9)
symbol_array = np.array(symbols)
inputs = to_vec(
    symbol_array
)  # converts the numpy array to an array of deuterium variables

kernel = to_vec(
    np.array(random_symbols(3, "k"))
)  # create a convolution kernel and conver to a variable

output = np.convolve(inputs, kernel) # convolve the inputs with the kernel
output.mean().backward() # gradients can be calculated only w.r.t. a scalar, hence the mean operation

print(inputs.flatten()[0].grad.simplify(rational=True))

>>> (1/12)*k_0 + (1/12)*k_1 + (1/12)*k_2

print(kernel.flatten()[0].grad.simplify(rational=True))

>>> (1/12)*x_0 + (1/12)*x_1 + (1/12)*x_2 + (1/12)*x_3 + (1/12)*x_4 + (1/12)*x_5 + (1/12)*x_6 + (1/12)*x_7 + (1/12)*x_8 + (1/12)*x_9

```

## Compilation

Deuterium supports two compilation modes: _just-in-time_ and _ahead-of-time_. 

### JIT compilation through substitution
This method is good for experimentation but slower in performance, hence better for experimentation. JIT compilation happens by substituting symbols in an expression for numeric values. Example:

```py

from deuterium import Variable, random_symbols, to_vec
import numpy as np

symbols = random_symbols(n=10, prefix="x")  # generate symbols (x_0,...,x_9)
symbol_array = np.array(symbols)
inputs = to_vec(
    symbol_array
)  # converts the numpy array to an array of deuterium variables

kernel = to_vec(
    np.array(random_symbols(3, "k"))
)  # create a convolution kernel and conver to a variable

output = np.convolve(inputs, kernel)
output.mean().backward()

print(inputs.flatten()[0].grad.subs({"k_0": 0.5, "k_1": 0.25, "k_2": 0.75, "k_3": 0.8})) #substitute symbols with numeric values into the gradient.

>>> 0.125

```

Notably, JITing can happen "partially", that is, it's possible to partially substitute numeric values and get back an expression which is more efficient to compute than the original one, but still contains some symbols. Example:


```py
from deuterium import Variable, random_symbols, to_vec
import numpy as np


symbols = random_symbols(n=10, prefix="x")  # generate symbols (x_0,...,x_9)
symbol_array = np.array(symbols)
inputs = to_vec(
    symbol_array
)  # converts the numpy array to an array of deuterium variables

kernel = to_vec(
    np.array(random_symbols(3, "k"))
)  # create a convolution kernel and conver to a variable

output = np.convolve(inputs, kernel)
output.mean().backward()

partial_grad = (
    inputs.flatten()[0].grad.subs({"k_0": 0.5, "k_3": 0.8}).simplify(rational=True)
)
print(partial_grad)

>>> 1/24 + (1/12)*k_1 + (1/12)*k_2 #partial symbolic expression

full_grad = partial_grad.subs({"k_1": 0.5, "k_2": 0.45}) #substitute the rest

print(full_grad)

>>> 0.120833333333333

```

### AOT compilation
It's also possible to compile an expression to a function which then behaves like a `numpy ufunc`, i.e. it broadcasts across the batch axis. The compiler requires a list of symbols which will become the inputs to the compiled function (i.e. the function signature), and can perform automatic common sub-expression elimination (with `cse=True`). If desired (and the LLVM tool-chain is installed), the `llvm` back-end can be specified, which takes longer to compile, but is faster to compute afterwards.

```py
from deuterium import random_symbols, to_vec
import numpy as np
import symengine as se

symbols = random_symbols(n=10, prefix="x")  # generate symbols (x_0,...,x_9)
symbol_array = np.array(symbols)
inputs = to_vec(
    symbol_array
)  # converts the numpy array to an array of deuterium variables

kernel = to_vec(
    np.array(random_symbols(3, "k"))
)  # create a convolution kernel and conver to a variable

output = np.convolve(inputs, kernel)
output.mean().backward()

grad = [i.grad for i in kernel.flatten()] # gradients w.r.t kernel

all_symbols = symbols + [s.data for s in kernel.flatten().tolist()] #variables in the function signature

gradient_func = se.Lambdify(all_symbols, grad, cse=True, backend="llvm") #compile with CSE and the LLVM back-end

values = np.random.random((4, len(all_symbols))) #random "batch" of inputs with size 4.

print(gradient_func(values)) #will print the gradient for every entry in the "batch"

>>> [[0.36386894 0.36386894 0.36386894]
 [0.31052365 0.31052365 0.31052365]
 [0.50394468 0.50394468 0.50394468]
 [0.40664184 0.40664184 0.40664184]]

```

Compiled functions can also be serialised, e.g. using `pickle`, for later use. This allows to pre-specify computations symbolically without any data and use them for model training later by loading the functions and passing the (private) data to them to obtain a result.

