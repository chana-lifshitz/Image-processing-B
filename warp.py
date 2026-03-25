import numpy as np

def warp_image(image: np.ndarray,
               angle_deg: float,
               scale_x: float,
               scale_y: float) -> np.ndarray:

    H, W, C = image.shape

    # TODO:
    # 1. Compute center (cx, cy)
    # 2. Build rotation matrix
    # 3. Build scaling matrix
    # 4. Compose full affine matrix
    # 5. Compute inverse
    # 6. For each output pixel:
    #       - map center coordinate backward
    #       - interpolate
    # 7. Return output image

def warp_image(image: np.ndarray,
               angle_deg: float,
               scale_x: float,
               scale_y: float) -> np.ndarray:

    H, W, C = image.shape

    # -------------------------------------------------
    # 1. center
    # -------------------------------------------------
    cx = W / 2
    cy = H / 2

    # -------------------------------------------------
    # 2. rotation matrix
    # -------------------------------------------------
    theta = np.deg2rad(angle_deg)

    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

    # -------------------------------------------------
    # 3. scaling matrix
    # -------------------------------------------------
    S = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])

    # -------------------------------------------------
    # 4. combine affine
    # -------------------------------------------------
    A = R @ S

    # translation matrices (for center)
    T1 = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ])

    T2 = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ])

    M = T2 @ A @ T1

    # -------------------------------------------------
    # 5. inverse matrix (for backward mapping)
    # -------------------------------------------------
    Minv = np.linalg.inv(M)

    # -------------------------------------------------
    # 6. output image
    # -------------------------------------------------
    output = np.zeros_like(image)

    for i in range(H):
        for j in range(W):

            # pixel center
            x = j + 0.5
            y = i + 0.5

            src = Minv @ np.array([x, y, 1])

            xs = src[0] - 0.5
            ys = src[1] - 0.5

            if xs < 0 or xs >= W-1 or ys < 0 or ys >= H-1:
                continue

            # -------------------------------------------------
            # 7. bilinear interpolation
            # -------------------------------------------------
            x0 = int(np.floor(xs))
            x1 = x0 + 1
            y0 = int(np.floor(ys))
            y1 = y0 + 1

            dx = xs - x0
            dy = ys - y0

            for c in range(C):
                v00 = image[y0, x0, c]
                v10 = image[y0, x1, c]
                v01 = image[y1, x0, c]
                v11 = image[y1, x1, c]

                value = (
                    v00 * (1-dx)*(1-dy) +
                    v10 * dx*(1-dy) +
                    v01 * (1-dx)*dy +
                    v11 * dx*dy
                )

                output[i, j, c] = np.clip(value, 0, 255)

    return output.astype(np.uint8)