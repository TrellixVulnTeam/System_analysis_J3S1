from PIL import Image
from pylab import *
import scipy.stats as st
import pandas as pd

file_path = '../resources/'


def img_transform(img_name):
    image = Image.open(file_path + img_name).convert('L')  # Открываем изображение и конвертируем в полутоновое
    image.save(file_path + "grayscale_" + img_name)
    image.show()


def print_histogram(img_name):
    img_array = array(Image.open(file_path + img_name))
    hist_data = img_array.flatten()

    figure()

    hist(hist_data, 10)

    figure()
    hist(hist_data, 128)
    show()

    stat = [np.mean(hist_data), np.std(hist_data), st.mode(hist_data)[0][0], np.median(hist_data)]
    print(stat)




img_transform("img1.jpg")
print_histogram("grayscale_img1.jpg")
