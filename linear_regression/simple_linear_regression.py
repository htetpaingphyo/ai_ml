"""
Simple Linear Regression Formula
ȳ = b0 + b1 * x
>> ȳ is the dependent variable
>> x is the independent variable
>> b0 is the ȳ-intercept
>> b1 is the slope
b1 = Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)
b0 = ȳ - b1 * x̄
"""

import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

x_bar = np.mean(x)
y_bar = np.mean(y)

# b1 = sum((x - x_bar)(y - y_bar)) / sum((x - x_bar)^2)
# b0 = y_bar - b1 * x_bar
b1 = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
b0 = y_bar - b1 * x_bar

y_hat = b0 + b1 * x

SST = np.sum((y - y_bar) ** 2)  # Total Sum of Square
SSR = np.sum((y_hat - y_bar) ** 2)  # Regression Sum of Square
SSE = np.sum((y - y_hat) ** 2)  # Error Sum of Square

R2 = SSR / SST

print("Fitted line: y_hat = {:.4f} + {:.4f} * x".format(b0, b1))
print("Predictions:", np.round(y_hat, 3))
print("SST = {:.4f}, SSR = {:.4f}, SSE = {:.4f}".format(SST, SSR, SSE))
print("R^2 = {:.4f}".format(R2))
