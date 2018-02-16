from sources._parser_ import Parser as _parser
from scipy import stats
import numpy as np
import math


file = "..\\resources\\winequality-red.txt"


# Доверительный интервал для мат.ожидания
def _conf_mean(x, conf_level=0.95):
    interval = stats.t.interval(conf_level, len(x)-1, loc=np.mean(x), scale=stats.sem(x))
    return interval


def _main_(file_name_):
    x = []
    data = _parser.parse(file_name_)

    for item in data[1:]:
        x.append(float(item[7]))

    mean_, variance_, std_ = stats.bayes_mvs(x)
    mean_value = mean_[0]
    mean_interval = mean_[1]
    variance_value = variance_[0]
    variance_interval = variance_[1]

_main_(file)