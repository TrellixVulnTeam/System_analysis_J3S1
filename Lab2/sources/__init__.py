from sources._parser_ import Parser as _parser
from scipy import stats
import matplotlib.pyplot as plt
import fpdf
import numpy as np
import math

file = "..\\resources\\winequality-red.txt"


# Доверительные интервалы мат.ожидания и дисперсии
def _conf_interval(x, conf_level=0.95):
    # Гистограмма
    hist, bins = np.histogram(x)
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(center, hist, align='center', width=width)
    plt.title('Histogram')
    plt.savefig('..\\resources\\hist.png')
    plt.show()

    # Мат.ожидание, дисперсия, стандартное отклонение
    mean_, variance_, std_ = stats.bayes_mvs(x, conf_level)

    return mean_[0], mean_[1], variance_[0], variance_[1]


def check_hypoth_with_unknown_variance(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    n_x = len(x)
    n_y = len(y)
    x_std = np.std(x)
    y_std = np.std(y)
    t = (x_mean - y_mean) / np.sqrt(
        ((n_x + n_y) / n_x * n_y) * (x_std ** 2 * (n_x - 1) + y_std ** 2 * (n_y - 1)) / (n_x + n_y - 2))

    t_test = stats.ttest_ind(np.asarray(x), np.asarray(y))

    print(t, t_test)


def make_pdf(mean, mean_interval, var, var_interval):
    _file = fpdf.FPDF()
    _file.add_page()
    _file.set_font("Arial", size=12)
    _file.cell(200, 10, 'Lab work #2', 0, 1, 'C')
    _file.cell(200, 10, 'Mean %f: ' % mean, 0, 1, 'L')
    _file.cell(200, 10, 'Mean`s interval: %f < %f < %f' % (mean_interval[0], mean, mean_interval[1]), 0, 1, 'L')
    _file.cell(200, 10, 'Variance: %g' % var, 0, 1, 'L')
    _file.cell(200, 10, 'Variance`s interval: %g < %g < %g' % (var_interval[0], var, var_interval[1]), 0, 1, 'L')
    _file.image('..\\resources\\hist.png', 0, 70, 200, 100)
    _file.output("../resources/result.pdf")


def _main_(file_name_):
    x = []
    y = []
    data = _parser.parse(file_name_)

    for item in data[1:]:
        x.append(float(item[7]))
        y.append(float(item[10]))

    # mean, mean_interval, var, var_interval = _conf_interval(x)
    # make_pdf(mean, mean_interval, var, var_interval)
    check_hypoth_with_unknown_variance(x, y)


_main_(file)
