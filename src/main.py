import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import mpmath
# extended / analytically continued Reimann Zeta Functions
# Euler-Maclaurin summation definition
def em_reimann_zeta(s, precision, sumterms=100):
    mpmath.mp.dps = precision
    return mpmath.nsum(lambda n: 1 / (n ** s), [1, mpmath.inf], method="euler-maclaurin", terms=sumterms)
# Pre-defined Zeta Function from mpmath lib
def mp_reimann_zeta(s, precision):
    mpmath.mp.dps = precision
    return mpmath.zeta(s)

if __name__ == "__main__":
    """
    xvals = [0.1 * i for i in range(1, 100)]
    yvals = [em_reimann_zeta(complex(0.5, x), 30).real for x in xvals]
    plt.plot(xvals, yvals, label="Reimann Zeta Function")
    plt.title("Reimann Zeta Function")
    plt.xlabel("Real Part(s)")
    plt.ylabel("Real Value")
    plt.legend()
    plt.show()
    """
    real_vals = np.linspace(-5, 5, 100)
    imag_vals = np.linspace(-5, 5, 100)
    real_part, imag_part = np.meshgrid(real_vals, imag_vals)
    zeta_values = np.vectorize(lambda x, y: em_reimann_zeta(complex(x, y), 30))(real_part, imag_part)
    #zeta_values = np.vectorize(lambda x, y: mp_reimann_zeta(complex(x, y), 30))(real_part, imag_part)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(real_part, imag_part, np.abs(zeta_values), cmap='viridis', alpha=0.8)
    ax.set_title('Analytically Continued Riemann Zeta Function')
    ax.set_xlabel('Real Part (s)')
    ax.set_ylabel('Imaginary Part (s)')
    ax.set_zlabel('Real Value')
    plt.show()
