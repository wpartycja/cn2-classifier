from collections import Counter
import pandas as pd
import numpy as np


class CN2:

    def __init__(self, star_max_size=3, epsilon=0.5):
        self.data = None
        self.star_max_size = star_max_size
        self.epsilon = epsilon  # significance (between 0 and 1) @TODO error handling when user inputs sth stupid
        self._P = []  # subset on which we will train in every iteration (when it's 0 algorithm is done)
        self._selectors = []  # set of atomic selectros

    # @TODO - here will appear all the magic
    def fit(self, dataset):
        pass

    # @TODO
    def predict(self):
        pass

    def find_selectors(self):
        """
        This function finds all possible selectors in given dataset
        writes to self._selectors in this pattern:
        self._selectors = [('column1', 'value1'), ('column1', 'value2')]
        """
        possible_selectors = self.data.drop(columns='class')

        for column in possible_selectors:
            possible_values = set(self.data[column])
            for value in possible_values:
                self._selectors.append((column, value))

    def specialize_star(self, star, selectors):
        """
        This function creates a new_star by doing intercetion with the previos
        star and then it removes non-valid complexes (with duplicated values)

        :param star: the previos star, list of complexes
        :param selectors: list of selectors which will specialize the star
        :return: new_star -> specialized star
        """
        new_star = []

        if len(star) == 0:
            for selector in selectors:
                new_star.append([selector])
        else:
            for complex in star:
                for selector in selectors:
                    new_complex = complex.copy()
                    new_complex.append(selector)

                    # checking if the not duplicate
                    duplicated = False
                    count = Counter([x[0] for x in new_complex])
                    for c_value in count.values():
                        if c_value != 1:
                            duplicated = True
                            break
                    if not duplicated:
                        new_star.append(new_complex)

        return new_star
    
    def most_common_class(self, data):
        """
        This function returns the most common class among all the examples given in input
        :param examples: DataFrame from which we want to find the most common class
        :return: the name of teh most commons class, count
        """
        most_common_class = data['class'].value_counts().head(1)
        return most_common_class.index[0], most_common_class[0]

    def get_covered_examples(self, data, best_complex):
        """
        This function searches for all examples that are covered by given complex
        :param data: DataFrame to search if the complex if covering examples from here or not
        :param complex: a list containign tuples (attribute, value)
        :return: indexes of covered examples
        """
        values_dict = {}
        for pair in best_complex:
            if pair[0] not in values_dict.keys():
                values_dict[pair[0]] = [pair[1]]
            else:
                values_dict[pair[0]].append(pair[1])
        
        for column in self.data.columns:
            if column not in values_dict:
                values_dict[column] = set(self.data[column])

        covered_examples = data[data.isin(values_dict).all(axis=1)]           
        return covered_examples.index

    def entropy(self, complex):
        """
        Calculates entopy of a complex
        :param complex: complex of which we are calculating the entropy
        :return: calculated entropy of the complex
        """
        covered_examples = self.get_covered_examples(self._P, complex) # self._P because here contrary to AQ algorithm we are training on the remainig examples
        classes = pd.DataFrame(self.data.iloc[covered_examples]['class'])
        classes_num = len(classes)
        class_count = classes.value_counts()
        class_prob = class_count / classes_num
        log = np.log2(class_prob)
        entropy = (class_prob * log).sum()

        return entropy * -1

    def salience(self, complex):  # istotność
        """
        calculating salience of the given complex
        """
        covered_ex = self.get_covered_examples(self._P, complex)
        classes = pd.DataFrame(self.data.iloc[covered_ex]['type'])
        classes_num = len(classes)
        class_count = classes.iloc[:,0].value_counts()
        class_prob = class_count.divide(classes_num)

        train_classes = self.data["type"]
        train_classes_num = len(train_classes)
        train_count = train_classes.value_counts()
        train_prob = train_count.divide(train_classes_num)

        return class_prob.multiply(np.log(class_prob.divide(train_prob))).sum() * 2


