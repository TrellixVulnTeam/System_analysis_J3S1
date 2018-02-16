from source.parser import Parser as pars
from source.grapher import Grapher as graph
from sklearn import linear_model
from scipy import stats
import numpy as np
import pandas as pd
import fpdf
import math

file = "..\\resources\\winequality-red.txt"


# Correlation coefficient
def correlation(x, y):
    coef = np.corrcoef(x, y)[1, 0]
    # Show correlation field
    graph.printCorrelField(x, y)
    return coef


# linear regression
def line_regression(x_list, y_list):
    regression = []
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    x = x.reshape(len(x), 1)
    y = y.reshape(len(y), 1)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    regression.append(regr.coef_.__float__())
    regression.append(regr.intercept_.__float__())
    regression.append(regr.score(x, y))

    graph.printRegresiionField(x, y, regr)
    return regression


def make_pdf(cor_coef, ttest, regr_coef, _interc):
    _file = fpdf.FPDF()
    _file.add_page()
    _file.set_font("Arial", size=12)
    _file.cell(200, 10, 'Lab work #1', 0, 1, 'C')
    _file.cell(200, 10, 'Correlation coefficient: ' + str(cor_coef), 0, 1, 'L')
    _file.cell(200, 10, 'Student`s t-statistic: ' + str(ttest), 0, 1, 'L')
    _file.cell(200, 10, 'Regression coefficients: ' + str(regr_coef) + ', ' + str(_interc), 0, 1, 'L')
    _file.cell(200, 10, 'Linear regression equation: Y = ' + str(regr_coef) + '* X + ' + str(_interc), 0, 1, 'L')
    _file.image('..\\resources\\lr.png', 0, 60)
    _file.output("../resources/result.pdf")


def create_table(data):
    table = pd.DataFrame(data[1:], columns=data[0])
    print(table)


def _main_(_file):
    x = []  # xList - list of density
    y = []  # yList - list of alcohol

    wine_data = pars.parse(file)

    for item in wine_data[1:]:
        x.append(float(item[7]))
        y.append(float(item[10]))

    cor_coef = correlation(x, y)
    reg_coef = line_regression(x, y)
    t_test = stats.ttest_ind(x, y)[0]
    make_pdf(cor_coef, t_test, reg_coef[0], reg_coef[1])
    create_table(wine_data)


_main_(file)
