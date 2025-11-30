# code from
# https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
# to fetch the dataset
# also install ucimlrepo package using pip intsll ucimlrepo
from ucimlrepo import fetch_ucirepo
import pandas as pd  # for data manipulation
import random

# fetch dataset
connectionist_bench_sonar_mines_vs_rocks = fetch_ucirepo(id=151)

# data (as pandas dataframes)
X = connectionist_bench_sonar_mines_vs_rocks.data.features
y = connectionist_bench_sonar_mines_vs_rocks.data.targets

# encode targets
pd.set_option('future.no_silent_downcasting', True)
y = y.replace({'R': 0, 'M': 1}).astype(int)

# merge features and targets into a single dataframe
dataset = pd.concat([X, y], axis=1).values.tolist()


# train a perceptron model weights
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] = weights[0] + l_rate * error  # bias
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
    return weights


# Perceptron Algorithm
def perceptron(train, test, l_rate, n_epoch):
    weights = train_weights(train, l_rate, n_epoch)
    predictions = []
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions


# make prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]  # bias
    return 1.0 if activation >= 0.0 else 0.0


# calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        # flatten all folds except current
        train_set = [row for f in folds if f != fold for row in f]
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def cross_validation_split(dataset, n_folds):
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = []
        while len(fold) < fold_size and dataset_copy:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def main():
    # merge features and targets into a single dataframe
    dataset = pd.concat([X, y], axis=1).values.tolist()
    n_folds = 3
    l_rate = 0.1
    n_epoch = 5
    scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


main()
