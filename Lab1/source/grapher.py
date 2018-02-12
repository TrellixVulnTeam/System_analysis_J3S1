import matplotlib.pyplot as plt
class Grapher:

    def printCorrelField(x, y):
        plt.axis('auto')

        plt.scatter(x, y, color='black')

        plt.xticks(())
        plt.yticks(())

        plt.show()
