from deuterium import Variable

# from sympy.abc import a, b, c
from symengine import symbols

a, b, c = symbols("a b c")


def test_symbols():
    _a = Variable(a)
    _b = Variable(b)
    _c = Variable(c)

    _i = _a - _b + _c

    _j = _i ** _b

    _k = _j.exp().sqrt()

    _l = 3 + _k / _i

    _m = _l.log()

    _d = _m * (_a ** -_b)

    _d.backward()

    assert (_a.grad.subs({a: 1, b: 2, c: 3}) + 2.97477174267099) < 1e-5
    assert (_b.grad.subs({a: 1, b: 2, c: 3}) + 0.0627510511750698) < 1e-5
    assert (_c.grad.subs({a: 1, b: 2, c: 3}) - 0.827809224675755) < 1e-5
