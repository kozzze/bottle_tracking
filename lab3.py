import cv2
import numpy as np

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    for x in range(size):
        for y in range(size):
            exponent = -((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)
            kernel[x, y] = np.exp(exponent)
    return kernel

def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    return kernel / np.sum(kernel)

def apply_custom_gaussian_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    height, width = image.shape
    k_size = kernel.shape[0]
    pad = k_size // 2
    result = np.zeros_like(image)

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            region = image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            result[i, j] = np.sum(region * kernel)

    return result

def main():
    img = cv2.imread("/Users/kozzze/Desktop/Учеба/bottle_tracking/img/1.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    configs = [
        (3, 0.5),
        (5, 1.0),
        (7, 1.5),
    ]

    for size, sigma in configs:
        kernel = gaussian_kernel(size, sigma)
        norm_kernel = normalize_kernel(kernel)

        blurred = apply_custom_gaussian_blur(img, norm_kernel)
        cv2.imwrite(f"custom_blur_size{size}_sigma{sigma}.jpg", blurred)

    # сравнение OpenCV
    opencv_blur = cv2.GaussianBlur(img, (7, 7), 1.5)
    cv2.imwrite("opencv_blur_7_1.5.jpg", opencv_blur)

if __name__ == "__main__":
    main()
