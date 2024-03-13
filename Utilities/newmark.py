import numpy as np

def Newmark(M, C, K, f, W, h, tf, a, b, n):
    # Define E matrices
    E1 = (1/h**2)*M + (b/h)*C + a*K
    E2 = (1/h**2)*M + (b/h)*C
    E3 = (1/h)*M + (b-a)*C
    E4 = (0.5 - a)*M + 0.5*h*(b - 2*a)*C

    # Time vector
    t = np.arange(0, tf+h, h)
    
    # Initialize x, dx, ddx
    x = np.zeros((n, len(t)))
    dx = np.zeros((n, len(t)))
    ddx = np.zeros((n, len(t)))
    
    # Initial conditions (you need to define x0, dx0, and ddx0)
    # x[:, 0] = x0
    # dx[:, 0] = dx0
    # ddx[:, 0] = np.linalg.solve(M, -C @ dx0 - K @ x0)

    # Newmark Integration
    for i in range(1, len(t)):
        print(t[i])
        F = a * np.sin(W * t[i]) * f
        x[:, i] = np.linalg.solve(E1, E2 @ x[:, i-1] + E3 @ dx[:, i-1] + E4 @ ddx[:, i-1] + F)
        ddx[:, i] = (1/(a*h**2)) * (x[:, i] - x[:, i-1]) - (1/(a*h)) * dx[:, i-1] + \
                    (1 - 1/(2*a)) * ddx[:, i-1]
        dx[:, i] = dx[:, i-1] + h * ((1 - b) * ddx[:, i-1] + b * ddx[:, i])

    return t, x, dx, ddx