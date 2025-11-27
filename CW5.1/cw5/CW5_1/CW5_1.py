import numpy as np

# Step 1: Generate data
data = np.random.normal(loc=50, scale=10, size=500)

complete_data = data

# Step 2: Create samples
sample_30 = np.random.choice(data, size=30, replace=False)
# ---->>>      # sample_30 = []
               # for i in range(30):
               #     choose = np.random.choice(data)
               #     sample_30.append(choose)


# Step 3: Calculate statistics for complete data
mean_complete_data = complete_data.mean()
median_complete_data = np.median(complete_data)
std_complete_data = complete_data.std()
var_complete_data = complete_data.var()

# Step 3: Calculate statistics for sample
mean_sample_30 = sample_30.mean()
median_sample_30 = np.median(sample_30)
std_sample_30 = sample_30.std()
var_sample_30 = sample_30.var()

# Step 4: Print results
print("=== COMPLETE DATA (500 points) ===")
print(f"Mean: {mean_complete_data:.2f}")
print(f"Median: {median_complete_data:.2f}")
print(f"Standard Deviation: {std_complete_data:.2f}")
print(f"Variance: {var_complete_data:.2f}")

print("\n=== SAMPLE DATA (30 points) ===")
print(f"Mean: {mean_sample_30:.2f}")
print(f"Median: {median_sample_30:.2f}")
print(f"Standard Deviation: {std_sample_30:.2f}")
print(f"Variance: {var_sample_30:.2f}")

print("\n=== COMPARISON WITH TRUE VALUES ===")
print(f"True Mean: 50, Sample Mean: {mean_sample_30:.2f}, Difference: {abs(50 - mean_sample_30):.2f}")
print(f"True STD: 10, Sample STD: {std_sample_30:.2f}, Difference: {abs(10 - std_sample_30):.2f}")
