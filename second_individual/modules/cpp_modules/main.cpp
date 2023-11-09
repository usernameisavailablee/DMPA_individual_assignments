#include <opencv2/opencv.hpp>

cv::Mat double_threshold_filtering(const cv::Mat& input_image, int low_th, int high_th) {
    cv::Mat output_image = input_image.clone();  // Создаем копию входного изображения

    int n = input_image.rows;
    int m = input_image.cols;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (input_image.at<uchar>(i, j) >= high_th || input_image.at<uchar>(i, j) <= low_th) {
                output_image.at<uchar>(i, j) = 0;
            } else {
                output_image.at<uchar>(i, j) = 255;
            }
        }
    }

    return output_image;
}

int main() {
    // Загрузка изображения
    cv::Mat input_image = cv::imread("../../data/bottle.jpg", cv::IMREAD_GRAYSCALE);

    if (input_image.empty()) {
        std::cerr << "Failed to load the image." << std::endl;
        return 1;
    }

    // Параметры для двойной бинаризации
    int low_threshold = 128;
    int high_threshold = 255;

    // Применение двойной бинаризации
    cv::Mat processed_image = double_threshold_filtering(input_image, low_threshold, high_threshold);

    // Сохранение отфильтрованного изображения
    cv::imwrite("output_image.jpg", processed_image);

    return 0;
}
