import numpy as np 
import matplotlib.pyplot as plt  


A = np.random.normal(0, np.sqrt(2), 100)

B = A.copy() 

indices = np.random.choice(100,10,replace=False)


for i in indices:
    B[i] = np.random.choice([-20, 20])



range_A = abs(np.min(A) - np.max(A)) 
std_A = np.std(A) 

range_B = abs(np.min(B) - np.max(B))
std_B = np.std(B)


print(f"Range A: {range_A:.2f}")
print(f"Range B: {range_B:.2f}")
print(f"Std Dev A: {std_A:.2f}") 
print(f"Std Dev B: {std_B:.2f}")



def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

x_a = np.linspace(np.min(A), np.max(A), 100) 
x_b = np.linspace(np.min(B),np.max(B),100) 


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(A, bins=20, alpha=0.7, color='blue', edgecolor='black', density=True)
plt.plot(x_a,gaussian(x_a,np.mean(A),np.std(A)),'r-') 
plt.axvline(np.mean(A), color='red', linestyle='--', label=f"mean: {np.mean(A):.2f}")
plt.title(f'Dataset A\nRange: {range_A:.1f}, Std: {std_A:.1f}')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2) 
plt.hist(B, bins=20, alpha=0.7, color='red', edgecolor='black', density=True)
plt.plot(x_b,gaussian(x_b,np.mean(B),np.std(B)),'b-') 
plt.axvline(np.mean(B), color='blue', linestyle='--', label=f"mean: {np.mean(B):.2f}")
plt.title(f'Dataset B\nRange: {range_B:.1f}, Std: {std_B:.1f}')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
