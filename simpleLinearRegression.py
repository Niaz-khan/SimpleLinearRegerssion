"""
Simple linear regression = intercept + slope (x)
Y = a +bX
slope(b) = (𝑁 * Σ(𝑋𝑌) − (Σ𝑋  ٭ Σ𝑌) / (𝑁 * Σ(X*X)−( Σ𝑋 * ΣX))
intercept = (ΣY − b(ΣX)) / N)
"""


class SimpleLinearRegression:
    def __init__(self):
        self.ind_X = None
        self.dep_y = None
        self.x_into_y = None
        self.double_x = None
        self.sum_of_x = None
        self.sum_of_y = None
        self.sum_of_x_into_y = None
        self.sum_of_double_x = None
        self.number_of_pairs = None
        self.slope = None
        self.intercept = None
        self.simple_regression = None

    def train(self, independent_x, dependent_y):
        """
        this method takes x, y and train the model
        x = independent colum as list
        y = dependent column as list
        :param independent_x:
        :param dependent_y:
        :return:
        """
        self.ind_X = independent_x
        self.dep_y = dependent_y

        self.x_into_y = [x * y for x, y in zip(self.ind_X, self.dep_y)]

        self.double_x = [pow(x, 2) for x in self.ind_X]

        self.sum_of_x = sum(self.ind_X)
        self.sum_of_y = sum(self.dep_y)
        self.sum_of_x_into_y = sum(self.x_into_y)
        self.sum_of_double_x = sum(self.double_x)
        self.number_of_pairs = len(self.x_into_y)

        # calculating the slope
        self.slope = (((self.number_of_pairs * self.sum_of_x_into_y) - (self.sum_of_x * self.sum_of_y)) /
                      (self.number_of_pairs * self.sum_of_double_x - pow(self.sum_of_x, 2)))

        self.intercept = (self.sum_of_y - self.slope * self.sum_of_x) / self.number_of_pairs

    def predict(self, independent_x):
        """
        this method predict dependent value for provided independent value
        :param independent_x:
        :return:
        """
        self.simple_regression = self.intercept + self.slope * independent_x
        print(f"model predict {self.simple_regression:.3f} for independent {independent_x}")


model = SimpleLinearRegression()
x = (60, 61, 62, 63, 65)
y = (3.1, 3.6, 3.8, 4, 4.1)
model.train(x, y)
model.predict(70)


# import numpy as np
#
# class SimpleLinearRegression:
#     def __init__(self):
#         self.ind_X = None
#         self.dep_y = None
#         self.slope = None
#         self.intercept = None
#
#     def train(self, independent_x, dependent_y):
#         if len(independent_x) != len(dependent_y):
#             raise ValueError("Input lists must be of the same length")
#
#         if (not all(isinstance(x, (int, float)) for x in independent_x) or
#                 not all(isinstance(y, (int, float)) for y in dependent_y)):
#             raise ValueError("Input lists must contain only numeric values")
#
#         self.ind_X = np.array(independent_x)
#         self.dep_y = np.array(dependent_y)
#
#         self.slope = np.dot(self.ind_X, self.dep_y) / np.dot(self.ind_X, self.ind_X)
#         self.intercept = np.mean(self.dep_y) - self.slope * np.mean(self.ind_X)
#
#     def predict(self, independent_x):
#         if not isinstance(independent_x, (int, float)):
#             raise ValueError("Input must be a numeric value")
#
#         return self.intercept + self.slope * independent_x
#
#     def evaluate(self):
#         predictions = self.intercept + self.slope * self.ind_X
#         mse = np.mean((predictions - self.dep_y) ** 2)
#         r2 = 1 - (np.sum((self.dep_y - predictions) ** 2) / np.sum((self.dep_y - np.mean(self.dep_y)) ** 2))
#         return mse, r2
#
# # Example usage:
# model = SimpleLinearRegression()
# x = [60, 61, 62, 63, 65]
# y = [3.1, 3.6, 3.8, 4, 4.1]
# model.train(x, y)
# print(model.predict(70))
# mse, r2 = model.evaluate()
# print(f"MSE: {mse}, R²: {r2}")
