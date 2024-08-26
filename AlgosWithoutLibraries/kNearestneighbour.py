"""
I am going to implement the KNN with sklearn lib
"""
from math import sqrt
import pandas as pd


def euclidean_distance(x2, x1, y2, y1):
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))


class KNearestNeighbourClassifier:
    def __init__(self):
        self.x1 = None
        self.x2 = None
        self.data_set = None
        self.k_values = None
        self.lst_of_distance = []

    def train(self, dicti, k):
        """
        this method takes dictionary as input and train model
        :param dicti, k:
        {'Points': ['A', 'B', 'C', 'D', 'E', 'P'],
        'x1': [1, 2, 3, 6, 7, 4],
        'x2': [2, 3, 3, 5, 8, 4],
        'Class': ['RED', 'RED', 'BLUE', 'BLUE', 'RED', '?']}
        :return:
        """
        self.data_set = dicti
        dataset = pd.DataFrame(self.data_set)
        # extracting the required columns
        self.x1 = dataset["x1"]
        self.x2 = dataset["x2"]
        # getting the required point's (x, y)
        x_2 = self.x1.iloc[-1]
        y_2 = self.x2.iloc[-1]

        # for x_1, y_1 in zip(x1[:-1], x2[:-1]):
        for x_1, y_1 in zip(self.x1, self.x2):
            distance = euclidean_distance(x_2, x_1, y_2, y_1)
            self.lst_of_distance.append(round(distance, 3))

        # round(list_of_distance[0], 2) # round any float digit to two decimal places
        self.data_set["Distance"] = self.lst_of_distance

        # ranking(indexing)
        self.data_set = dataset.sort_values(by="Distance").reset_index(drop=True)

        # extracting K values
        self.k_values = self.data_set["Class"][1:k]

        # creating a dictionary that will contain all classes
        majority_value = {}
        for value in self.k_values:
            if value in majority_value:
                majority_value[value] += 1
            else:
                majority_value[value] = 1

        print(majority_value)

        # getting the most repeat value(class)
        m_r_value = None
        max_count = 0
        for value, count in majority_value.items():
            if count > max_count:
                max_count = count
                m_r_value = value
        print(m_r_value)

        # assigning the class
        dataset["Class"][0] = m_r_value

        # indexing the df
        self.data_set = self.data_set.sort_values(by="Points").reset_index(drop=True)
        print(self.data_set)


model = KNearestNeighbourClassifier()
model.train({'Points': ['A', 'B', 'C', 'D', 'E', 'P'], 'x1': [1, 2, 3, 6, 7, 4],
             'x2': [2, 3, 3, 5, 8, 4], 'Class': ['RED', 'RED', 'BLUE', 'BLUE', 'RED', '?']}, k=3)

