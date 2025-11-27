import numpy as np
import matplotlib.pyplot as plt

# 1- Generate random data with normal distribution
data = np.random.normal(loc=0, scale=1, size=1000)
data = data[(data >= -3) & (data <= 3)]  # Limit between -3 and 3

print(f"Number of data points after filtering: {len(data)}")

# Gaussian function for comparison
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

x = np.linspace(-3, 3, 100)

# 2- Create visualizations
plt.figure(figsize=(15, 5))

# 2-a: Histogram with bins=10
plt.subplot(1, 3, 1)
plt.hist(data, bins=10, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.plot(x, gaussian(x, 0, 1), 'r-', linewidth=2)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram with bins=10')
plt.grid(True, alpha=0.3)

# 2-b: Histogram with bins=50
plt.subplot(1, 3, 2)
plt.hist(data, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
plt.plot(x, gaussian(x, 0, 1), 'r-', linewidth=2)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram with bins=50')
plt.grid(True, alpha=0.3)

# 2-c: Boxplot
plt.subplot(1, 3, 3)
plt.boxplot(data)
plt.ylabel('Value')
plt.title('Boxplot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3- Analysis
print("\n=== DATA ANALYSIS ===")
print(f"Mean: {np.mean(data):.3f}")
print(f"Standard Deviation: {np.std(data):.3f}")
print(f"Median: {np.median(data):.3f}")
print(f"Minimum: {np.min(data):.3f}")
print(f"Maximum: {np.max(data):.3f}")
