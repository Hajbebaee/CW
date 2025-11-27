
import numpy as np
import matplotlib.pyplot as plt


population = np.random.uniform(low=0, high=10, size=10000)


sample_means_30 = [] 
sample_means_2 = []   


for i in range(500):
    
    sample_30 = np.random.choice(population, size=30, replace=False)
    mean_30 = np.mean(sample_30)
    sample_means_30.append(mean_30)
    
    
    sample_2 = np.random.choice(population, size=2, replace=False)
    mean_2 = np.mean(sample_2)
    sample_means_2.append(mean_2)


def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

x_30 = np.linspace(np.min(sample_means_30),np.max(sample_means_30),100)
x_2 = np.linspace(np.min(sample_means_2),np.max(sample_means_2),100)

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.hist(sample_means_30, bins=20, color='skyblue', edgecolor='black', alpha=0.7,density=True)

plt.axvline(np.mean(sample_means_30), color='red', linestyle='--', label=f'Mean: {np.mean(sample_means_30):.2f}')

plt.plot(x_30,gaussian(x_30,np.mean(sample_means_30),np.std(sample_means_30)),'r-')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.title('CLT: n=30 (Normal Distribution)')
plt.legend()
plt.grid(True, alpha=0.3)


plt.subplot(1, 2, 2)
plt.hist(sample_means_2, bins=20, color='lightcoral', edgecolor='black', alpha=0.7,density=True)

plt.axvline(np.mean(sample_means_2), color='red', linestyle='--', label=f'Mean: {np.mean(sample_means_2):.2f}')

plt.plot(x_2,gaussian(x_2,np.mean(sample_means_2),np.std(sample_means_2)),'r-')

plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.title('CLT: n=2 (Non-Normal)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5- Statistical analysis
print("=== CENTRAL LIMIT THEOREM DEMONSTRATION ===")
print(f"Population mean: {np.mean(population):.3f}")
print(f"Population standard deviation: {np.std(population):.3f}")
print("\n--- Sample Means Analysis ---")
print(f"n=30 - Mean of means: {np.mean(sample_means_30):.3f}")
print(f"n=30 - Std of means: {np.std(sample_means_30):.3f}")
print(f"n=2 - Mean of means: {np.mean(sample_means_2):.3f}")
print(f"n=2 - Std of means: {np.std(sample_means_2):.3f}")
