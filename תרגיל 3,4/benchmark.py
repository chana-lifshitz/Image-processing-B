import numpy as np
import time
from warp_ex4 import bilinear 
from warp_ex3 import warp_image_loop  

def measure(func, img, angle=30, sx=1.2, sy=1.2, runs=3):
    times = []

    for _ in range(runs):
        start = time.time()
        func(img, angle, sx, sy)
        end = time.time()
        times.append(end - start)

    return np.mean(times)

sizes = [
    (100, 100),
    (300, 300),
    (500, 500),
    (800, 800)
]

print(f"{'H':>6} {'W':>6} {'Loop (s)':>12} {'NumPy (s)':>12}")
print("-" * 40)

for h, w in sizes:
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    t_loop = measure(warp_image_loop, img)
    t_vec = measure(bilinear, img)

    print(f"{h:6} {w:6} {t_loop:12.4f} {t_vec:12.4f}")