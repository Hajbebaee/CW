import numpy as np 
import matplotlib.pyplot as plt  

data_1 = np.random.normal(-2, 1, 500) 
data_2 = np.random.normal(5, 1.5, 500) 

combine_data = np.concatenate([data_1, data_2])

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

x_1 = np.linspace(np.min(data_1), np.max(data_1), 100)
x_2 = np.linspace(np.min(data_2), np.max(data_2), 100)

plt.figure(figsize=(10, 6)) 


plt.hist(data_1, bins=20, density=True, color='blue', alpha=0.7, edgecolor='black', label='Data 1')
plt.plot(x_1, gaussian(x_1, np.mean(data_1), np.std(data_1)), 'r-', linewidth=2)

plt.hist(data_2, bins=20, density=True, color='red', alpha=0.7, edgecolor='black', label='Data 2 ')
plt.plot(x_2, gaussian(x_2, np.mean(data_2), np.std(data_2)), 'b-', linewidth=2)

plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Two Gaussian Distributions Combined')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Mean of combined data: {np.mean(combine_data):.3f}")
print(f"Mean of first data: {np.mean(data_1):.3f}") 
print(f"Mean of second data: {np.mean(data_2):.3f}") 
print(f"Calculated mean: {(np.mean(data_1) + np.mean(data_2)) / 2:.3f}")
print(f"Variance of combined data: {np.var(combine_data):.3f}")
