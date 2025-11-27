import numpy as np
import matplotlib.pyplot as plt


class_a = np.random.normal(0, np.sqrt(1), 200)
class_b = np.random.normal(3, np.sqrt(0.5), 200)
class_c = np.random.normal(-3, np.sqrt(2), 200)


def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


x_range = np.linspace(-8, 8, 1000)


y_a = gaussian(x_range, 0, 1)
y_b = gaussian(x_range, 3, np.sqrt(0.5))
y_c = gaussian(x_range, -3, np.sqrt(2))


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(class_a, bins=20, alpha=0.7, color='blue', density=True, label='Class A')
plt.hist(class_b, bins=20, alpha=0.7, color='black', density=True, label='Class B')
plt.hist(class_c, bins=20, alpha=0.7, color='red', density=True, label='Class C')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram of Classes')
plt.legend()
plt.grid(True, alpha=0.3)


plt.subplot(1, 3, 2)
plt.plot(x_range, y_a, 'blue', linewidth=2, label='Class A (mean=0, var=1)')
plt.plot(x_range, y_b, 'black', linewidth=2, label='Class B (mean=3, var=0.5)')
plt.plot(x_range, y_c, 'red', linewidth=2, label='Class C (mean=-3, var=2)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Gaussian Distributions')
plt.legend()
plt.grid(True, alpha=0.3)


plt.subplot(1, 3, 3)
new_point = 1.0
prob_a = gaussian(new_point, 0, 1)
prob_b = gaussian(new_point, 3, np.sqrt(0.5))
prob_c = gaussian(new_point, -3, np.sqrt(2))

classes = ['Class A', 'Class B', 'Class C']
probabilities = [prob_a, prob_b, prob_c]
colors = ['blue', 'black', 'red']

plt.bar(classes, probabilities, color=colors, alpha=0.7)
plt.axhline(y=max(probabilities), color='green', linestyle='--', alpha=0.8)
plt.ylabel('Probability')
plt.title(f'Classification for x={new_point}')
plt.grid(True, alpha=0.3)


winning_index = np.argmax(probabilities)
plt.text(winning_index, max(probabilities) + 0.01, 'WINNER', 
         ha='center', va='bottom', fontweight='bold', color='green')

plt.tight_layout()
plt.show()


print("Classification Results:")
print(f"Class A probability: {prob_a:.4f}")
print(f"Class B probability: {prob_b:.4f}")
print(f"Class C probability: {prob_c:.4f}")
print(f"Data point x={new_point} belongs to {classes[winning_index]}")
