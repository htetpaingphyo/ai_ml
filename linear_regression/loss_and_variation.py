"""
This program is to calculate loss and variation in linear regression using numpy and pandas.
 - SST (Total Sum Square) = Σ(yi - ȳ)²
 - SSR (Regression Sum of Squares) = Σ(ȳ - ŷi)²
 - SSE (Error Sum of Squares) = Σ(yi - ŷi)²
 - SST = SSR + SSE
"""

import numpy as np
import pandas as pd

x = np.array([2, 3, 4, 5], dtype=np.float32)
y = np.array([1, 2, 3, 4], dtype=np.float32)

# ŷ = b0 + b1 * x1 (fit simple linear regression using numpy)
b1, b0 = np.polyfit(x, y, 1)  # return [slope, intercept]
y_hat = b0 + b1 * x

y_bar = np.mean(y)  # declare ȳ
residual = y - y_hat

SSR = np.sum((y_hat - y_bar) ** 2)  # Regression Sum of Squares
SSE = np.sum((y - y_hat) ** 2)  # Error Sum of Squares
SST = SSR + SSE  # Total Sum of Squares
R2 = SSR / SST if SST != 0 else np.nan  # R squared

df = pd.DataFrame(
    {
        "x": x,
        "y (actual)": y,
        "y_hat (predicted)": np.round(y_hat, 4),
        "residual (y - y_hat)": np.round(residual, 4),
        "(y - y_bar)^2": np.round((y - y_bar) ** 2, 4),
        "(y_hat - y_bar)^2": np.round((y_hat - y_bar) ** 2, 4),
        "(y - y_hat)^2": np.round((y - y_hat) ** 2, 4),
    }
)

print(df, end="\n\n")
print("Fitted line: y_hat = {:.4f} + {:.4f} * x".format(b0, b1))
print("y_bar (mean of y): {:.4f}".format(y_bar))
print()
print("SST (Total)       = {:.4f}".format(SST))
print("SSR (Explained)   = {:.4f}".format(SSR))
print("SSE (Unexplained) = {:.4f}".format(SSE))
print()
print("Check: SST ≈ SSR + SSE -> {:.4f} ≈ {:.4f}".format(SST, SSR + SSE))
print("R^2 = SSR / SST   = {:.4f}".format(R2))
