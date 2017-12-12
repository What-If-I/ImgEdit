import numpy as np
from math import cos, pi


def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def idft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.divide(np.exp(2j * np.pi * k * n / N), N)
    return np.dot(M, x)


def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N & (N - 1) != 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:
        return dft(x)
    else:
        x_even = fft(x[::2])
        x_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([x_even + factor[:N / 2] * x_odd,
                               x_even + factor[N / 2:] * x_odd])


# def dct(x):
#     N = x.size
#     xk = np.zeros(N)
#     for k in range(N):
#         for n in range(N):
#             xn = x[n]
#             xk[k] += xn * np.cos(np.pi / N * (n + 1 / 2.0) * k)
#     return xk


def fft2(img):
    return map(fft, img)

#DCT


def f(u, v):
    if u + v % 2:
        return 0
    return 1


def sum(x, y):
    return x + y


def sigma(r, f):
    return reduce(sum, map(f, r))


def C(u):
    if u == 0:
        return 2 ** -0.5
    return 1.0


def F(u, v, N):
    return round((C(u) * C(v) / 4) * sigma(range(N), lambda i: sigma(range(N), lambda j: cos(
        (2 * i + 1) * u * pi / (2 * N)) * cos((2 * j + 1) * v * pi / (2 * N)) * f(i, j))), 2)


def dct(img):
    N = img.size
    return map(lambda u: map(lambda v: F(u, v, N), range(N)), range(N))
