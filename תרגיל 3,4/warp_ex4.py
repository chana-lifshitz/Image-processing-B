import numpy as np
# Q1
def nearest_neighbor(image: np.ndarray,
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
    # 5. inverse (backward mapping)
    # -------------------------------------------------
    Minv = np.linalg.inv(M)

    # -------------------------------------------------
    # 6. vectorized pixel grid
    # -------------------------------------------------
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # pixel centers
    x = x + 0.5
    y = y + 0.5

    ones = np.ones_like(x)

    coords = np.stack([x, y, ones], axis=0)   # (3, H, W)
    coords_flat = coords.reshape(3, -1)       # (3, H*W)

    # -------------------------------------------------
    # 7. backward mapping
    # -------------------------------------------------
    src = Minv @ coords_flat

    xs = src[0, :] - 0.5
    ys = src[1, :] - 0.5

    # -------------------------------------------------
    # 8. nearest neighbor
    # -------------------------------------------------
    xs_nn = np.round(xs).astype(int)
    ys_nn = np.round(ys).astype(int)

    # valid indices
    valid = (
        (xs_nn >= 0) & (xs_nn < W) &
        (ys_nn >= 0) & (ys_nn < H)
    )

    # destination indices
    x_dst = coords_flat[0, :].astype(int)
    y_dst = coords_flat[1, :].astype(int)

    x_dst = x_dst[valid]
    y_dst = y_dst[valid]

    x_src = xs_nn[valid]
    y_src = ys_nn[valid]

    # -------------------------------------------------
    # 9. build output
    # -------------------------------------------------
    output = np.zeros_like(image)

    output[y_dst, x_dst] = image[y_src, x_src]

    return output.astype(np.uint8)



#####תרגיל 2

def bilinear(image: np.ndarray,
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
    # 5. inverse (backward mapping)
    # -------------------------------------------------
    Minv = np.linalg.inv(M)

    # -------------------------------------------------
    # 6. יצירת גריד וקטורי
    # -------------------------------------------------
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # מרכז הפיקסל
    x = x + 0.5
    y = y + 0.5

    coords = np.stack([x, y, np.ones_like(x)], axis=0)
    coords_flat = coords.reshape(3, -1)

    # -------------------------------------------------
    # 7. backward mapping
    # -------------------------------------------------
    src = Minv @ coords_flat

    xs = src[0, :] - 0.5
    ys = src[1, :] - 0.5

    # -------------------------------------------------
    # 8. bilinear interpolation
    # -------------------------------------------------
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)

    x1 = x0 + 1
    y1 = y0 + 1

    dx = xs - x0
    dy = ys - y0

    # רק פיקסלים חוקיים
    valid = (
        (x0 >= 0) & (x1 < W) &
        (y0 >= 0) & (y1 < H)
    )

    x0 = x0[valid]
    x1 = x1[valid]
    y0 = y0[valid]
    y1 = y1[valid]

    dx = dx[valid]
    dy = dy[valid]

    x_dst = coords_flat[0, :].astype(int)[valid]
    y_dst = coords_flat[1, :].astype(int)[valid]

    # -------------------------------------------------
    # דגימת הפיקסלים
    # -------------------------------------------------
    v00 = image[y0, x0]
    v10 = image[y0, x1]
    v01 = image[y1, x0]
    v11 = image[y1, x1]

    # התאמה למימד צבע
    dx = dx[:, None]
    dy = dy[:, None]

    # נוסחת bilinear
    values = (
        v00 * (1 - dx) * (1 - dy) +
        v10 * dx * (1 - dy) +
        v01 * (1 - dx) * dy +
        v11 * dx * dy
    )

    # -------------------------------------------------
    # יצירת תמונת פלט
    # -------------------------------------------------
    output = np.zeros_like(image, dtype=np.float32)
    output[y_dst, x_dst] = values

    return np.clip(output, 0, 255).astype(np.uint8)