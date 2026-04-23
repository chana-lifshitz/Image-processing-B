import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import correlate2d

#Q1
# א. kernel
def initialize_kernel():
    return np.array([
        [-1,  2,  1],
        [-2,  1, -3],
        [ 3,  0, -1]
    ], dtype=np.float32)


# ב. תמונה
def get_image():
    return np.array([
        [103, 102, 101, 100],
        [104, 103, 102, 101],
        [ 53,  52,  51,  50],
        [ 45,  53,  52,  51]
    ], dtype=np.uint8)


# ג. מימוש עם לולאות
def cross_correlate_loop(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape

    out_H = H - kH + 1
    out_W = W - kW + 1

    result = np.zeros((out_H, out_W), dtype=np.float32)

    for i in range(out_H):
        for j in range(out_W):
            patch = image[i:i+kH, j:j+kW]
            result[i, j] = np.sum(patch * kernel)

    return result


# ד. מימוש עם numpy (vectorized)
def cross_correlate_np(image, kernel):
    windows = sliding_window_view(image, kernel.shape)
    # windows.shape = (2, 2, 3, 3)

    result = np.sum(windows * kernel, axis=(2, 3))
    return result.astype(np.float32)


# ה. מימוש עם scipy
def cross_correlate_scipy(image, kernel):
    result = correlate2d(image, kernel, mode='valid')
    return result.astype(np.float32)


# ו. השוואה
def compare_cross_correlations():
    image = get_image()
    kernel = initialize_kernel()

    r1 = cross_correlate_loop(image, kernel)
    r2 = cross_correlate_np(image, kernel)
    r3 = cross_correlate_scipy(image, kernel)

    return np.allclose(r1, r2) and np.allclose(r2, r3)


# בדיקה (לא חובה)
if __name__ == "__main__":
    print("Loop result:\n", cross_correlate_loop(get_image(), initialize_kernel()))
    print("NumPy result:\n", cross_correlate_np(get_image(), initialize_kernel()))
    print("SciPy result:\n", cross_correlate_scipy(get_image(), initialize_kernel()))
    print("All equal:", compare_cross_correlations())

  