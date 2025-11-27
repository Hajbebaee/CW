from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data[:, 0]  # Sepal Length
Y = 2 * X + 5


X_mean, Y_mean = np.mean(X), np.mean(Y)
X_std, Y_std = np.std(X), np.std(Y)
X_range, Y_range = np.max(X)-np.min(X), np.max(Y)-np.min(Y)

print("=== Comparison X vs Y ===")
print(f"X - Mean: {X_mean:.2f}, Std: {X_std:.2f}, Range: {X_range:.2f}")
print(f"Y - Mean: {Y_mean:.2f}, Std: {Y_std:.2f}, Range: {Y_range:.2f}")

# Create Y

# Gaussian function
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Create x values for Gaussian curves
x_values = np.linspace(np.min(X), np.max(X), 100)
y_values = np.linspace(np.min(Y), np.max(Y), 100)

# Create visualization
plt.figure(figsize=(15, 5))

# Plot X
plt.subplot(1, 3, 1)
plt.hist(X, bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.plot(x_values, gaussian(x_values, np.mean(X), np.std(X)), 'red', linewidth=2, label='Gaussian')
plt.axvline(X_mean, color='red', linestyle='--', label=f'Mean: {X_mean:.2f}')
plt.xlabel('Sepal Length (X)')
plt.ylabel('Density')
plt.title('Original Data (X)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Y
plt.subplot(1, 3, 2)
plt.hist(Y, bins=20, density=True, alpha=0.7, color='green', edgecolor='black')
plt.plot(y_values, gaussian(y_values, np.mean(Y), np.std(Y)), 'red', linewidth=2, label='Gaussian')
plt.axvline(Y_mean, color='red', linestyle='--', label=f'Mean: {Y_mean:.2f}')
plt.xlabel('Transformed Data (Y)')
plt.ylabel('Density')
plt.title('Transformed Data (Y)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot both together for comparison
plt.subplot(1, 3, 3)
plt.hist(X, bins=20, density=True, alpha=0.5, color='blue', edgecolor='black', label='X')
plt.hist(Y, bins=20, density=True, alpha=0.5, color='green', edgecolor='black', label='Y')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('X vs Y Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
