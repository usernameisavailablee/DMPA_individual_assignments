import numpy as np


class OurGaussianBlur:
  @classmethod
  def gaussian_filter(cls, filter_shape, sigma):
    m, n = filter_shape
    m_half = m // 2
    n_half = n // 2

    gaussian_filter = np.zeros((m, n), np.float32)
    sum_of_el = 0
    for y in range(-m_half, m_half + 1):
      for x in range(-n_half, n_half + 1):
        first_part = 1 / (2.0 * np.pi * sigma ** 2.0)
        exp_term = np.exp(-(x ** 2.0 + y ** 2.0) / (2.0 * sigma ** 2.0))
        gaussian_filter[y + m_half, x + n_half] = first_part * exp_term
        sum_of_el += first_part * exp_term
    for y in range(-m_half, m_half + 1):
      for x in range(-n_half, n_half + 1):
        gaussian_filter[y + m_half, x + n_half] /= sum_of_el
    return gaussian_filter

  def my_filter(img, kernel):
    try:
      img_height, img_width, img_canals = img.shape
    except:
      img_height, img_width = img.shape
      img_canals = 1

    kernel_height, kernel_width = kernel.shape

    result_height = img_height - kernel_height + 1
    result_width = img_width - kernel_width + 1

    result = np.zeros((result_height, result_width), dtype=np.float32)

    if img_canals != 1:
      for i in range(result_height):
        for j in range(result_width):
          for canal in range(img_canals):
            result[i, j, canal] = np.sum(img[i:i + kernel_height, j:j + kernel_width, canal] * kernel)
    else:
      for i in range(result_height):
        for j in range(result_width):
          result[i, j] = np.sum(img[i:i + kernel_height, j:j + kernel_width] * kernel)
    return result
