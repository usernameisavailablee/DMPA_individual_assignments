import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class PrintImg:
  @classmethod
  def print_n_m_imgs(cls, rows, tables, titles, imgs):
    for k in range(rows * tables):
      plt.subplot(rows, tables, k + 1), plt.imshow(imgs[k], 'gray')
      plt.title(titles[k])
      plt.xticks([]), plt.yticks([])
    plt.show()