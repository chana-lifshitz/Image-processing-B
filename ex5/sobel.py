  #Q3
import sys
import os
import numpy as np
import cv2


# נורמליזציה ל־0–255
def normalize_to_uint8(img):
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.uint8)

    norm = (img - min_val) / (max_val - min_val)
    norm = norm * 255
    return norm.astype(np.uint8)


def main():
    if len(sys.argv) < 2:
        print("Usage: python sobel.py image.jpg")
        return

    image_path = sys.argv[1]

    # קריאת תמונה
    image = cv2.imread(image_path)

    if image is None:
        print("Failed to load image")
        return

    # --- א. grayscale ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- kernels של Sobel ---
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    # --- convolution ---
    gx = cv2.filter2D(gray, cv2.CV_32F, sobel_x)
    gy = cv2.filter2D(gray, cv2.CV_32F, sobel_y)

    # --- ב. |Gx| + נורמליזציה ---
    gx_abs = np.abs(gx)
    gx_norm = normalize_to_uint8(gx_abs)

    # --- ג. |Gy| + נורמליזציה ---
    gy_abs = np.abs(gy)
    gy_norm = normalize_to_uint8(gy_abs)

    # --- ד. magnitude ---
    magnitude = np.sqrt(gx**2 + gy**2)
    mag_norm = normalize_to_uint8(magnitude)

    # --- שמירת קבצים ---
    # base, ext = os.path.splitext(image_path)
    base, _ = os.path.splitext(image_path)
    ext = ".jpg"
    cv2.imwrite(base + "_grayscale" + ext, gray)
    cv2.imwrite(base + "_gx" + ext, gx_norm)
    cv2.imwrite(base + "_gy" + ext, gy_norm)
    cv2.imwrite(base + "_magnitude" + ext, mag_norm)

    print("Done!")


if __name__ == "__main__":
    main()