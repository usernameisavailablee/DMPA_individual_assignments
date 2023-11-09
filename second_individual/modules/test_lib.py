import sys
import numpy as np
import cv2 as cv

# Загрузка изображения
image = cv.imread('../data/bottle.jpg', cv.IMREAD_COLOR)

# Преобразование в оттенки серого
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray,(7,7),5)
# Бинаризация изображения
_, thresh = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)

# Поиск контуров
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)

# Создание пустого изображения для отображения только контуров
contour_image = np.zeros_like(image)

# Рисование контуров на пустом изображении
cv.drawContours(contour_image, contours, -1, (255, 255, 255), 2)

# Отображение изображения с контурами
cv.imshow('Contours Only', contour_image)
cv.waitKey(0)
cv.destroyAllWindows()
