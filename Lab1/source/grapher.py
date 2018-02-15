import matplotlib.pyplot as plt


class Grapher:
    def printCorrelField(X, Y):
        plt.axis('auto')
        plt.scatter(X, Y, s=10)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def printRegresiionField(X, Y, regr):
        plt.axis('auto')
        plt.scatter(X, Y, color='black', s=10)
        plt.plot(X, regr.predict(X), color='blue', linewidth=3)
        plt.xlabel('density')
        plt.ylabel('alcohol')
        plt.savefig('../resources/lr.png')
        plt.show()
