import numpy as np
import matplotlib.pyplot as plt


# Function to calculate Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function for Gradient Descent to update slope (m) and intercept (c)
def gradient_descent(x, y, m, c, learning_rate):
    N = len(y)
    y_pred = m * x + c
    # Calculate gradients
    dm = (-2/N) * np.sum(x * (y - y_pred))
    dc = (-2/N) * np.sum(y - y_pred)
    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc
    return m, c

# Load dataset (replace 'Nairobi_Office_Price.csv' with your actual file path)
data = np.genfromtxt('Nairobi Office Price Ex.csv', delimiter=',', skip_header=1)
x = data[:, 0]  # Office size
y = data[:, 1]  # Office price

# Initialize parameters
m, c = np.random.rand(), np.random.rand()
learning_rate = 0.01
epochs = 10

# Training loop
for epoch in range(epochs):
    y_pred = m * x + c
    error = mean_squared_error(y, y_pred)
    print(f'Epoch {epoch+1}: Mean Squared Error = {error:.4f}')
    # Update m and c
    m, c = gradient_descent(x, y, m, c, learning_rate)

# Plot the line of best fit after training
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, m * x + c, color='red', label='Line of best fit')
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.legend()
plt.title('Linear Regression: Office Size vs Price')
plt.show()

# Prediction for office size of 100 sq. ft
office_size = 100
predicted_price = m * office_size + c
print(f'Predicted office price for 100 sq. ft: {predicted_price:.2f}')
