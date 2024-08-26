class SimpleLinearRegression:
    def __init__(self):
        self.ind_x = None
        self.dep_y = None
        self.x_into_y = None
        self.double_x = None
        self.sum_of_ind_x = None
        self.sum_of_dep_y = None
        self.sum_of_x_into_y = None
        self.sum_of_double_x = None
        self.number_of_pairs = None
        self.slope = None
        self.intercept = None
        self.regression = None

    def train_the_model(self, independent_x, dependent_y):
        """
        this method map the model from independent column to dependent column provided in list both separatly
        """
        # assign the lists to vars
        self.ind_x = independent_x
        self.dep_y = dependent_y

        # multiplying the x and y
        self.x_into_y = [x * y for x, y in zip(self.ind_x, self.dep_y)]

        # product of x and x
        self.double_x = [x ** 2 for x in self.ind_x]

        # sums of all vars
        self.sum_of_ind_x = sum(self.ind_x)
        self.sum_of_dep_y = sum(self.dep_y)
        self.sum_of_x_into_y = sum(self.x_into_y)
        self.sum_of_double_x = sum(self.double_x)
        self.number_of_pairs = len(self.x_into_y)

        # finding the slope
        self.slope = ((self.number_of_pairs * self.sum_of_x_into_y) - (self.sum_of_ind_x * self.sum_of_dep_y)) / (
                    (self.number_of_pairs * self.sum_of_double_x) - (self.sum_of_ind_x ** 2))

        # print(self.sum_of_dep_y - slope * self.sum_of_ind_x)
        # print((self.sum_of_dep_y - slope * self.sum_of_ind_x) / self.number_of_pairs)

        # finding the intercept
        self.intercept = (self.sum_of_dep_y - self.slope * self.sum_of_ind_x) / self.number_of_pairs
        # print(round(self.intercept))

    def predict(self, independent_x):
        """
        this method predict a dependent value for provided independent value
        """
        self.regression = self.intercept + self.slope * independent_x
        print(f"for The independent {independent_x} machine predict dependent {self.regression:.2f}".title())


model = SimpleLinearRegression()
model.train_the_model((60, 61, 62, 63, 65), (3.1, 3.6, 3.8, 4, 4.1))
model.predict(70)

