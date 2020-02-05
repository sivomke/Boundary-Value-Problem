import numpy as np
import matplotlib.pyplot as plt
import boundary_problem as bp


n = 31
h = (bp.b - bp.a)/n
grid = np.array([bp.a + i*h for i in range(n+1)])


def a(i: int) -> float:
    return bp.k(grid[i] - 0.5*h)


def b(i: int) -> float:
    return bp.p(grid[i] - 0.5*h)


def phi(i: int) -> float:
    return bp.f(grid[i])


def d(i: int) -> float:
    return bp.q(grid[i])


alpha_1 = bp.alpha_1 + 0.5*h*d(0)
alpha_2 = bp.alpha_2 + 0.5*h*d(n)
mu_1 = bp.mu_1 + 0.5*h*phi(0)
mu_2 = bp.mu_2 + 0.5*h*phi(n)

A = np.zeros((n+1, n+1))
A[0, 0] = a(1)/h - 0.5*b(1) + alpha_1
A[0, 1] = -a(1)/h + 0.5*b(1)
A[n, n-1] = -a(n)/h - 0.5*b(n)
A[n, n] = a(n)/h + 0.5*b(n) + alpha_2
for i in range(1, n):
    A[i, i-1] = -a(i)/h - 0.5*b(i)
    A[i, i] = a(i)/h + a(i+1)/h + 0.5*b(i) - 0.5*b(i+1) + h*d(i)
    A[i, i + 1] = -a(i+1)/h + 0.5*b(i+1)

# print(A)


f = np.zeros(n+1)
f[0] = mu_1
for i in range(1, n):
    f[i] = phi(i)*h
f[n] = mu_2

y_grid = np.array(np.linalg.solve(A, f))
for i in range(n+1):
    print("{grid_func:10.8f}  {target:10.8f}  {error:10.8f}".format(grid_func = y_grid[i], target = bp.u(grid[i]),
                                                                  error = abs(y_grid[i] - bp.u(grid[i]))))


# plotting target func and its numeric approx
def comparison(grid_function, target, grid):
    target_val = [target(x) for x in grid]
    plt.plot(grid, grid_function, label='grid function')
    plt.plot(grid, target_val, label='target function')
    plt.legend()
    plt.show()


comparison(y_grid, bp.u, grid)
