import numpy as np

# Age data of individuals
ages = np.array([25, 30, 45, 50])

# Whether or not they have the disease
has_disease = np.array([True, False, True, True])

# Select the ages of individuals with the disease
ages_with_disease = ages[has_disease]

# Calculate the true average age
true_avg_age = np.mean(ages_with_disease)
print(f'True Average Age: {true_avg_age}')

# Define a function to add Laplacian noise
def add_laplace_noise(data, sensitivity, epsilon):
    return data + np.random.laplace(loc=0, scale=sensitivity/epsilon)

# Sensitivity for the query (max age - min age)
sensitivity = np.max(ages) - np.min(ages)

# Privacy budget
epsilon = 0.5

# Calculate the differentially private average age
private_avg_age = add_laplace_noise(true_avg_age, sensitivity, epsilon)
print(f'Private Average Age: {private_avg_age}')
