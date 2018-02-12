from source.script import Parser as parser
from source.grapher import Grapher as grapher
import numpy as np

file = "..\\winequality-red.txt"

# xList - list of density
# yList - list of alcohol
xList = []
yList = []

wineData = parser.parse(file)

for item in wineData[1:]:
    xList.append(float(item[7]))
    yList.append(float(item[10]))

coef = np.corrcoef(xList, yList)[1, 0]
print(coef)

grapher.printCorrelField(xList, yList)