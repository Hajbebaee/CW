import numpy as np
import matplotlib.pyplot as plt

# Generate x values from -10 to 10
x = np.linspace(start=-10, stop=10, num=1000)
mu = np.mean(x)  # Center of the distribution

def gaussian_function(x, mu, sigma):
    """Calculate Gaussian function"""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Create plot
plt.figure(figsize=(10, 6))

# Plot Gaussian curves for different sigma values
plt.plot(x, gaussian_function(x, mu, 0.5), label=' 0.5', linewidth=2)
plt.plot(x, gaussian_function(x, mu, 1), label=' 1', linewidth=2)
plt.plot(x, gaussian_function(x, mu, 3), label=' 3', linewidth=2)

# Add labels and styling
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gaussian Function for Different  Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
