
#Q2
# gradient_descent_z.py
import sys
import math

# LEARNING_RATE = 0.1
LEARNING_RATE = 2
NUM_ITERATIONS = 1000

def z(x, y):
    return math.sin(x) + math.sin(y)

def grad_z(x, y):
    dz_dx = math.cos(x)
    dz_dy = math.cos(y)
    return dz_dx, dz_dy

def gradient_descent(x, y):

    print(
        f"Starting point:  x={x:.4f},  y={y:.4f},  z={z(x, y):.4f}"
    )

    for _ in range(NUM_ITERATIONS):
        dz_dx, dz_dy = grad_z(x, y)

        x = x - LEARNING_RATE * dz_dx
        y = y - LEARNING_RATE * dz_dy

    print(
        f"Minimum found:   x={x:.4f},  y={y:.4f},  z={z(x, y):.4f}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python gradient_descent_z.py <x> <y>")
        sys.exit(1)

    x = float(sys.argv[1])
    y = float(sys.argv[2])

    gradient_descent(x, y)

#Q3
    # data_normalization.py

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# טעינת הנתונים
data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

print("=== Before Normalization ===\n")

for column in X.columns:
    mean = X[column].mean()
    std = X[column].std()

    print(f"{column}")
    print(f"Mean = {mean:.4f}")
    print(f"Std  = {std:.4f}")
    print()

# Normalization
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_normalized = pd.DataFrame(
    X_normalized,
    columns=X.columns
)

print("\n=== After Normalization ===\n")

for column in X_normalized.columns:
    mean = X_normalized[column].mean()
    std = X_normalized[column].std()

    print(f"{column}")
    print(f"Mean = {mean:.4f}")
    print(f"Std  = {std:.4f}")
    print()