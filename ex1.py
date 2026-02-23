import numpy as np
import matplotlib.pyplot as plt
import math

####q1
def degrees_to_radians(deg):
    return deg * math.pi / 180

degrees_list = [0, 90, 180, 45, 30, 10, 5, 1]

print("degrees,radians,sin,cos")

for deg in degrees_list:
    rad = degrees_to_radians(deg)
    sin_val = math.sin(rad)
    cos_val = math.cos(rad)
    print(f"{deg},{rad},{sin_val},{cos_val}")


###q3
theta = math.radians(30)

r_30 = np.array([
    [math.cos(theta), -math.sin(theta)],
    [math.sin(theta),  math.cos(theta)]
])

print("r_30 =")
print(r_30)

sx_2 = np.array([
    [2, 0],
    [0, 1]
])

print("\nsx_2 =")
print(sx_2)

rs = r_30 @ sx_2
print("\nrs = r_30 @ sx_2 =")
print(rs)

sr = sx_2 @ r_30
print("\nsr = sx_2 @ r_30 =")
print(sr)

rectangle = np.array([
    [-1, -0.5],
    [ 1, -0.5],
    [ 1,  0.5],
    [-1,  0.5],
    [-1, -0.5]  
]).T

rect_rotated = r_30 @ rectangle
rect_scaled = sx_2 @ rectangle
rect_sr = sr @ rectangle
rect_rs = rs @ rectangle

plt.figure(figsize=(8, 8))

plt.plot(rectangle[0], rectangle[1], label="Original")
plt.plot(rect_rotated[0], rect_rotated[1], label="Rotated")
plt.plot(rect_scaled[0], rect_scaled[1], label="Scaled X2")
plt.plot(rect_sr[0], rect_sr[1], label="SR = sx_2 @ r_30")
plt.plot(rect_rs[0], rect_rs[1], label="RS = r_30 @ sx_2")

plt.axhline(0)
plt.axvline(0)
plt.axis("equal")
plt.legend()
plt.title("Rotation and Scaling of Rectangle")
plt.show()