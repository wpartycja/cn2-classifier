from collections import Counter


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
        This function creates a new_star by doing intercetion with the previos star
        and then it removes non-valid complexes (with duplicated values)

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
        Return the most common class among all the examples given in input
        :param examples: DataFrame from which we want to find the most common class
        :return: the name of teh most commons class, count
        """
        most_common_class = data['class'].value_counts().head(1)
        
        return most_common_class.indes[0], most_common_class[0]