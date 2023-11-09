import sys
import numpy as np
import cv2 as cv

def double_threshold_filtering(img, low_pr, high_pr):
    down = low_pr
    up = high_pr
    n, m = img.shape
    for i in range(n):
        for j in range(m):
            if (img[i, j] >= up) or (img[i, j] <= down):
                img[i, j] = 255
            else:
                img[i, j] = 0
    return img

def find_boundary_pixels(binary_image):
    h, w = binary_image.shape

    # Создаем пустое изображение той же формы, где граничные пиксели будут белыми
    boundary_image = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Проверяем пиксель и его восемь соседей
            neighbors = [binary_image[i-1, j-1], binary_image[i-1, j], binary_image[i-1, j+1],
                         binary_image[i, j-1], binary_image[i, j], binary_image[i, j+1],
                         binary_image[i+1, j-1], binary_image[i+1, j], binary_image[i+1, j+1]]

            # Если хотя бы один из соседей имеет другое состояние, то это граничный пиксель
            if 255 in neighbors and 0 in neighbors:
                boundary_image[i, j] = 255

    return boundary_image



image = cv.imread('../data/bottle.jpg', cv.IMREAD_COLOR)


gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray,(7,7),5)

thresh = double_threshold_filtering(gray,128,255)


boundary_pixels = find_boundary_pixels(thresh)


cv.imwrite("boundary_image.jpg", boundary_pixels)

cv.imshow('thresh',thresh)
cv.imshow('boundary',boundary_pixels)
cv.waitKey(0)
cv.destroyAllWindows()
