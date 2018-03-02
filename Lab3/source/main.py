from PIL import Image
from pylab import *
import scipy.stats as st
import scipy.signal as sign
import numpy as np

file_path = '../resources/'


# Transform RGB image to graysclae image
def img_transform(img_name):
    image = Image.open(file_path + img_name).convert('L')  # Открываем изображение и конвертируем в полутоновое
    image.save(file_path + "grayscale_" + img_name)
    image.show()
    return image


# Image histogram and statistic
def img_histogram(image):
    img_array = array(image)
    hist_data = img_array.flatten()

    figure()
    hist(hist_data, 10)
    show()

    figure()
    hist(hist_data, 128)
    show()

    stat = [np.mean(hist_data), np.std(hist_data), st.mode(hist_data)[0][0], np.median(hist_data)]
    return hist_data, stat


# Image correlation
def img_correlation(img1, img2):
    x = np.array(img1)
    y = np.array(img2)

    return sign.correlate2d(x, y)


def hyp_check():
    print()


def main():
    image1 = img_transform("im1.jpg")
    image2 = img_transform("im2.jpg")

    hist1, hist_stat1 = img_histogram(image1)
    hist2, hist_stat2 = img_histogram(image2)
    print("First image statistic: ", hist_stat1)
    print("Second image statistic: ", hist_stat2)

    hist_correlation_coef = np.corrcoef(np.asarray(hist1).flatten(), np.asarray(hist2).flatten())[1, 0]
    print("Histogram correlation: ", hist_correlation_coef)

    img_correl1 = img_correlation(image1, image2)
    print(hist_correlation_coef, '\n', img_correl1)


if __name__ == "__main__":
    main()
