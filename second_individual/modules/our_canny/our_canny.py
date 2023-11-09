import cv2 as cv
import numpy as np

from modules.our_canny.our_gaussian_blur import OurGaussianBlur
from modules.our_canny.print_img import PrintImg


class OurCanny:

  @classmethod
  def start(cls, path, filter_shape, sigma, low_pr, high_pr):
    img = cls.prepare_img(path, filter_shape, sigma)
    gradient_magnitude, gradient_angle = cls.search_for_gradients(img)
    suppressed = cls.non_max_suppression(gradient_magnitude, gradient_angle)
    resulf_of_method = cls.double_threshold_filtering(suppressed, low_pr, high_pr)

    PrintImg.print_n_m_imgs(1, 2, [
      "prepared image.\nfilter_shape: " + str(filter_shape[1]) + " " + str(filter_shape[0]) + ". sigma: " + str(sigma),
      "result_of_method\nlow_pr: " + str(low_pr) + ", high_pr: " + str(high_pr)],
                            [img, resulf_of_method])

  # take_img_from_path_convert_to_grey_and_blur
  @classmethod
  def prepare_img(cls, path, filter_shape, sigma):
    try:
      img = cv.imread(path)
    except:
      img = cv.imread('other\d_noiseImgLamp.jpg')

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernal = OurGaussianBlur.gaussian_filter(filter_shape, sigma)
    return OurGaussianBlur.my_filter(img, kernal)

  @classmethod
  def get_sobel_operator_x(cls):
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

  @classmethod
  def get_sobel_operator_y(cls):
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

  # generate 2 matrix matrix of length values andmatrix of gradient angle values: result is gradient vector at each point
  @classmethod
  def search_for_gradients(cls, img):
    gradient_x = OurGaussianBlur.my_filter(img, cls.get_sobel_operator_x())
    gradient_y = OurGaussianBlur.my_filter(img, cls.get_sobel_operator_y())

    # PrintImg.print_n_m_imgs(1, 2, ["gradient_x", "gradient_y"],
    #                         [gradient_x, gradient_y])

    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_angle = (np.tan(gradient_x, gradient_y))
    gradient_angle_not_my = np.round((np.arctan2(gradient_x, gradient_y)) / (np.pi / 4) * (np.pi / 4) - (np.pi / 2), 0)

    n, m = gradient_x.shape

    for i in range(n):
      for j in range(m):
        x = gradient_x[i, j]
        y = gradient_y[i, j]
        tg = gradient_angle[i, j]
        gradient_angle[i, j] = cls.classify_gradient(x, y, tg)

    return gradient_magnitude, gradient_angle

  @classmethod
  def classify_gradient(cls, gradient_x, gradient_y, tg):
    if (gradient_x > 0 and gradient_y < 0 and tg < -2.414) or (gradient_x < 0 and gradient_y < 0 and tg > 2.414):
      return 0
    elif gradient_x > 0 and gradient_y < 0 and tg < -0.414:
      return 1
    elif (gradient_x > 0 and gradient_y < 0 and tg > -0.414) or (gradient_x > 0 and gradient_y > 0 and tg < 0.414):
      return 2
    elif gradient_x > 0 and gradient_y > 0 and tg < 2.414:
      return 3
    elif (gradient_x > 0 and gradient_y > 0 and tg > 2.414) or (gradient_x < 0 and gradient_y > 0 and tg < -2.414):
      return 4
    elif gradient_x < 0 and gradient_y > 0 and tg < -0.414:
      return 5
    elif (gradient_x < 0 and gradient_y > 0 and tg > -0.414) or (gradient_x < 0 and gradient_y < 0 and tg < 0.414):
      return 6
    elif gradient_x < 0 and gradient_y < 0 and tg < 2.414:
      return 7
    else:
      return -1

  @classmethod
  def non_max_suppression(cls, gradient_magnitude, gradient_angle):
    height, width = gradient_magnitude.shape
    suppressed = np.copy(gradient_magnitude)

    gradient_angle = (gradient_angle * 180.0 / np.pi) % 180

    for i in range(1, height - 1):
      for j in range(1, width - 1):
        angle = gradient_angle[i, j]

        if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
          neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
        elif (22.5 <= angle < 67.5):
          neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
        elif (67.5 <= angle < 112.5):
          neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
        else:
          neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]

        if gradient_magnitude[i, j] < max(neighbors):
          suppressed[i, j] = 0
    return suppressed

  @classmethod
  def double_threshold_filtering(cls, img, low_pr, high_pr):
    down = low_pr * 255
    up = high_pr * 255

    n, m = img.shape
    clone_of_img = np.copy(img)
    for i in range(n):
      for j in range(m):
        if clone_of_img[i, j] >= up:
          clone_of_img[i, j] = 255
        elif clone_of_img[i, j] <= down:
          clone_of_img[i, j] = 0
        else:
          clone_of_img[i, j] = 127
    return clone_of_img
