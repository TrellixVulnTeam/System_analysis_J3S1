from sources._parser_ import Parser as _parser
from scipy import stats
import matplotlib.pyplot as plt
import fpdf
import numpy as np


file = "..\\resources\\winequality-red.txt"


# Доверительные интервалы мат.ожидания и дисперсии
def _conf_interval(x, conf_level=0.95):
    # Гистограмма
    hist, bins = np.histogram(x)
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:])/2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(center, hist, align='center', width=width)
    ax.set_xticks(bins)
    # plt.savefig('..\\resources\\hist.png')
    plt.show()

    mean_, variance_, std_ = stats.bayes_mvs(x, conf_level)

    mean_value = mean_[0]
    mean_interval = mean_[1]
    variance_value = variance_[0]
    variance_interval = variance_[1]

    return mean_value, mean_interval, variance_value, variance_interval


def make_pdf(mean, mean_interval, var, var_interval):
    _file = fpdf.FPDF()
    _file.add_page()
    _file.set_font("Arial", size=12)
    _file.cell(200, 10, 'Lab work #2', 0, 1, 'C')
    _file.cell(200, 10, 'Mean %f: ' % mean, 0, 1, 'L')
    _file.cell(200, 10, 'Mean`s interval: %f < %f < %f' % (mean_interval[0], mean, mean_interval[1]), 0, 1, 'L')
    _file.cell(200, 10, 'Variance: %g' % var, 0, 1, 'L')
    _file.cell(200, 10, 'Variance`s interval: %g < %g < %g' % (var_interval[0], var, var_interval[1]), 0, 1, 'L')
    _file.add_page('L')
    _file.image('..\\resources\\hist.png', 0, 0)
    _file.output("../resources/result.pdf")


def _main_(file_name_):
    x = []
    data = _parser.parse(file_name_)

    for item in data[1:]:
        x.append(float(item[7]))

    mean, mean_interval, var, var_interval = _conf_interval(x)
    make_pdf(mean, mean_interval, var, var_interval)

_main_(file)