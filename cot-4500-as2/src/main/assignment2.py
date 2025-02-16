import numpy as np

# Number 1, Neville's method
def neville_interpolation(x_points, y_points, x):
    n = len(x_points)
    Q = [[0] * n for _ in range(n)]

    for i in range(n):
        Q[i][0] = y_points[i]

    for j in range(1, n):
        for i in range(n - j):
            Q[i][j] = ((x - x_points[i + j]) * Q[i][j - 1] + (x_points[i] - x) * Q[i + 1][j - 1]) / (x_points[i] - x_points[i + j])

    return Q[0][n - 1]

x_points = [3.6, 3.8, 3.9]
y_points = [1.675, 1.436, 1.318]

x = 3.7
result = neville_interpolation(x_points, y_points, x)
print(result,end = "\n")
print()

# Number 2, Newton's forward method

def newton_forward_coefficients(x, y):
    n = len(x)
    coef = np.zeros(n)
    coef[0] = y[0]

    for i in range(1, n):
        y[:n - i] = (y[1:n - i + 1] - y[:n - i]) / (x[i:n] - x[:n - i])
        coef[i] = y[0]
    return coef


def newton_forward_interpolation(x, coefficients, x_value):
    n = len(x)
    h = x[1] - x[0] 
    p = (x_value - x[0]) / h

    result = coefficients[0]
    term = 1

    for i in range(1, n):
        term *= (p - (i - 1)) / i
        result += coefficients[i] * term

    return result

x = np.array([7.2, 7.4, 7.5, 7.6])
y = np.array([23.5492, 25.3913, 26.8224, 27.4589])

coefficients = newton_forward_coefficients(x, y.copy())
print("Coefficients:")
for coef in coefficients[1:]:
    print(coef)

# Number 3
x_value = 7.3
approximation = newton_forward_interpolation(x, coefficients, x_value)
print("\nf(7.3):")
print(approximation)
print()

# Number 4, Hermite Polynomials

def hermite_interpolation(x, f, df):
    n = len(x)
    z = np.zeros(2 * n)
    Q = np.zeros((2 * n, 2 * n))

    for i in range(n):
        z[2 * i] = z[2 * i + 1] = x[i]
        Q[2 * i][0] = Q[2 * i + 1][0] = f[i]
        Q[2 * i + 1][1] = df[i]

        if i != 0:
            Q[2 * i][1] = (Q[2 * i][0] - Q[2 * i - 1][0]) / (z[2 * i] - z[2 * i - 1])

    for j in range(2, 2 * n):
        for i in range(j, 2 * n):
            Q[i][j] = (Q[i][j - 1] - Q[i - 1][j - 1]) / (z[i] - z[i - j])

    return z, Q

x = np.array([3.6, 3.8, 3.9])
f = np.array([1.675, 1.436, 1.318])
df = np.array([-1.195, -1.188, -1.182])

z, Q = hermite_interpolation(x, f, df)

for i in range(len(z)):
    row = [f"{z[i]:.6e}"] + [f"{Q[i, j]:.6e}" for j in range(5)]
    print("[", " ".join(row), "]")
print()

# Number 5, Cublic Spline interpolation

x = np.array([2,5,8,10])
f_x = np.array([3,5,7,9])

h = np.diff(x)

n = len(x)

A = np.zeros((n, n))

A[0, 0] = 1
A[-1, -1] = 1

for i in range(1, n - 1):
    A[i, i - 1] = h[i - 1]
    A[i, i] = 2 * (h[i - 1] + h[i])
    A[i, i + 1] = h[i]

b = np.zeros(n)
for i in range(1, n - 1):
    b[i] = 3 * ((f_x[i + 1] - f_x[i]) / h[i] - (f_x[i] - f_x[i - 1]) / h[i - 1])

x_spline = np.linalg.solve(A,b)

np.set_printoptions(precision=7, suppress=True)

print("A:")
for row in A:
    print(row)

print("b:",b)

print("x:" ,x_spline)

