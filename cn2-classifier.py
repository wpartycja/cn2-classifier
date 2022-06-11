import json
from collections import Counter
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import hamming
from tqdm import tqdm


class CN2:

    def __init__(self, star_max_size=3, epsilon=0.5):
        self.data = None
        self.star_max_size = star_max_size
        self.epsilon = epsilon if 0 < epsilon < 1 else 0.5  # significance
        self._P = []  # subset on which we will train in every iteration (when it's empty algorithm is done)
        self._selectors = []  # set of atomic selectors

    def fit(self, dataset) -> list:
        self.data = pd.read_csv(dataset)
        self._P = self.data.copy()
        self.find_selectors()

        rules = []
        classes = pd.DataFrame(self.data['class'])
        classes_count = classes.value_counts()

        while not self._P.empty:
            print(f"{len(self._P)} examples left")
            best_complex = self.calculate_best_complex()
            if best_complex is None:
                break
            covered_examples = self.get_covered_examples(self._P, best_complex)
            most_common_class, count = self.most_common_class(covered_examples)
            self._P.drop(covered_examples, inplace=True)

            total = classes_count[most_common_class] if most_common_class in classes_count.keys() else 0
            coverage = count / total
            precision = count / len(covered_examples)

            rules.append((best_complex, most_common_class, coverage, precision))

        most_common_class, count = self.most_common_class(self.data.index)
        total = classes_count[most_common_class]
        coverage = count / total
        precision = count / len(self.data)
        rules.append(([], most_common_class, coverage, precision))

        return rules

    def predict(self, test_file_path, rules) -> list[dict[str, int | float | None | Any]]:
        test_data = pd.read_csv(test_file_path)
        test_classes = test_data.iloc[:, -1]
        test_data = test_data.drop(columns='class')
        predicted_classes = ["" for _ in range(len(test_data))]
        rules_performance = []
        remaining_examples = test_data.copy()

        for rule in rules:
            rule_complex = rule[0]

            if rule_complex is not None:
                covered_examples = self.get_covered_examples(remaining_examples, rule_complex)
                remaining_examples.drop(covered_examples, inplace=True)
                indexes = list(covered_examples)
            elif len(remaining_examples) > 0:
                indexes = list(remaining_examples.index)
            else:
                continue

            predicted_class = rule[1]
            correct_predictions, wrong_predictions = 0, 0

            for index in indexes:
                predicted_classes[index] = predicted_class
                if test_classes[index] == predicted_class:
                    correct_predictions += 1
                else:
                    wrong_predictions += 1

            total = correct_predictions + wrong_predictions
            accuracy = correct_predictions / total if total > 0 else None

            performance = {
                'rule': rule,
                'predicted classes': predicted_class,
                'correct predictions': correct_predictions,
                'wrong predictions': wrong_predictions,
                'accuracy': accuracy
            }
            rules_performance.append(performance)

        return rules_performance#, hamming(predicted_classes, test_classes)  # * len(predicted_classes)

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

    def most_common_class(self, covered_ex):
        """
        This function returns the most common class among all the examples given in input
        :param examples: DataFrame from which we want to find the most common class
        :return: the name of teh most commons class, count
        """
        most_common_class = self.data.iloc[covered_ex, :]['class'].value_counts().head(1)
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
        covered_examples = self.get_covered_examples(self._P,
                                                     complex)  # self._P because here contrary to AQ algorithm we are training on the remainig examples
        classes = pd.DataFrame(self.data.iloc[covered_examples]['class'])
        classes_num = len(classes)
        class_count = classes.iloc[:, 0].value_counts()
        class_prob = class_count / classes_num
        log = np.log2(class_prob)
        entropy = (class_prob * log).sum()

        return entropy * -1

    def salience(self, complex):  # istotność
        """
        calculating salience of the given complex
        """
        covered_ex = self.get_covered_examples(self._P, complex)
        classes = pd.DataFrame(self.data.iloc[covered_ex]['class'])
        classes_num = len(classes)
        class_count = classes.iloc[:, 0].value_counts()
        class_prob = class_count.divide(classes_num)

        train_classes = self.data["class"]
        train_classes_num = len(train_classes)
        train_count = train_classes.value_counts()
        train_prob = train_count.divide(train_classes_num)

        return class_prob.multiply(np.log(class_prob.divide(train_prob))).sum() * 2

    def calculate_best_complex(self):
        best_complex = None
        best_entropy = np.inf
        best_salience = 0
        star = []

        while True:
            all_entropies = {}
            new_star = self.specialize_star(star, self._selectors)

            # needs to be this way because list is not a hashable type (so here can't be complex)
            for idx in tqdm(range(len(new_star))):
                complex = new_star[idx]
                cpx_salience = self.salience(complex)
                if cpx_salience > self.epsilon:
                    entropy = self.entropy(complex)
                    all_entropies[idx] = entropy
                    if entropy < best_entropy:
                        best_complex = complex.copy()
                        best_entropy = entropy
                        best_salience = cpx_salience

            best_complexes = sorted(all_entropies.items(), key=lambda x: x[1], reverse=False)[:self.star_max_size]

            star = [new_star[x[0]] for x in best_complexes]
            # for cpx in best_complexes:
            #     star.append(new_star[cpx[0]])

            if len(star) == 0 or best_salience < self.epsilon:
                break
            print(f"Best salience: {best_salience}; Star size: {len(star)}")

        return best_complex


def pretty_print_results(results: list) -> None:
    print("\n\n")
    for result in results:
        rule = result["rule"]
        correct = result["correct predictions"]
        wrong = result["wrong predictions"]
        if correct + wrong == 0:
            continue

        print(f"For rule:\n{rule}")
        print(f"Correct: {correct}; Wrong: {wrong}")
        print(f"Accuracy: {correct / (correct + wrong) * 100}%")
        # print(f"{rule}: {correct} correct, {wrong} wrong")
        print("\n")
    return


def iris_test():
    cn2 = CN2()
    rules = cn2.fit('./data/csv/iris.csv')
    # print(rules)
    results = cn2.predict('./data/csv/iris.csv', rules)
    # print(acc)
    # print(perf)
    pretty_print_results(results)
    exit(1)
    with open("./data/iris_report.json", "w") as f:
        json.dump(results, f)

    exit(1)
    # print(f"Accuracy: {acc}")
    keys, vals = [], []

    for data in perf:
        val = []
        for key, value in data.items():
            keys.append(key)
            val.append(value)
        vals.append(val)

    exit(1)

    table = pd.DataFrame([v for v in vals], columns=list(dict.fromkeys(keys)))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(table)
    table.to_csv('./data/output/iris_performance.csv', index=False)


if __name__ == '__main__':
    iris_test()
    # cn2 = CN2()
    # cn2.fit("./data/csv/iris.csv")
    # print(cn2.calculate_best_complex())
