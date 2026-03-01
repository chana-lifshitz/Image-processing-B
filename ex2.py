import numpy as np
import math
####Q1
#2
def translation_matrix(a, b):
    return np.array([
        [1, 0, a],
        [0, 1, b],
        [0, 0, 1]
    ])

T = translation_matrix(5, -1)
print("מטריצת ההזזה 3×3:")
print(T)

#3
def rotation_matrix(theta):
    rad = math.radians(theta)
    return np.array([
        [math.cos(rad), -math.sin(rad), 0],
        [math.sin(rad),  math.cos(rad), 0],
        [0,              0,             1]
    ])
print("\nQ3 – Rotation Matrix (θ=90°):")
R = rotation_matrix(90)
R_clean = np.round(R, decimals=0)  # עיגול לאפסים ול־1
print(R_clean)

#4
def scale_matrix(sx, sy=None):
    if sy is None:
        sy = sx
    return np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ])
print("\nQ4 – Scale Matrix (sx=2, sy=3):")
print(scale_matrix(2, 3))
print("\nQ4 – Uniform Scale Matrix (sx=2):")
print(scale_matrix(2))

#5
def rotate_around_point(theta, x, y):
    T1 = translation_matrix(-x, -y)
    R  = rotation_matrix(theta)
    T2 = translation_matrix(x, y)
    return T2 @ R @ T1

print("\nQ5 – Rotation 30° around point (100,200):")
print(rotate_around_point(30, 100, 200))

####Q2
import numpy as np
import matplotlib.pyplot as plt


def rotation_matrix(theta_deg):
    theta = np.radians(theta_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def scale_matrix(sx, sy):
    return np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ])


rectangle = np.array([
    [-1, -0.5, 1],
    [ 1, -0.5, 1],
    [ 1,  0.5, 1],
    [-1,  0.5, 1],
    [-1, -0.5, 1]  
]).T  
rect_original = rectangle

R30 = rotation_matrix(30)
rect_rot30 = R30 @ rectangle

R45 = rotation_matrix(45)
Sx2 = scale_matrix(2, 1)
rect_rot45_scale = Sx2 @ (R45 @ rectangle)
rect_scale_rot45 = R45 @ (Sx2 @ rectangle)

plt.figure(figsize=(8,8))

plt.plot(rect_original[0], rect_original[1], label="Original")
plt.plot(rect_rot30[0], rect_rot30[1], label="Rotate 30°")
plt.plot(rect_rot45_scale[0], rect_rot45_scale[1], label="Rotate 45° then Scale X2")
plt.plot(rect_scale_rot45[0], rect_scale_rot45[1], label="Scale X2 then Rotate 45°")

plt.axhline(0)
plt.axvline(0)
plt.gca().set_aspect('equal', 'box')
plt.legend()
plt.title("Rectangle Transformations (Homogeneous Coordinates)")
plt.grid(True)
plt.show()

#####Q3
def linear_interpolation(I0, I1, t):
    return (1 - t) * I0 + t * I1


def bilinear_interpolation(I00, I01, I10, I11, alpha, beta):
    return (
        (1 - alpha) * (1 - beta) * I00 +
        alpha * (1 - beta) * I01 +
        (1 - alpha) * beta * I10 +
        alpha * beta * I11
    )

I00 = 10
I01 = 20
I10 = 30
I11 = 40

alpha = 0.5
beta = 0.5

result = bilinear_interpolation(I00, I01, I10, I11, alpha, beta)
print("Interpolated value:", result)