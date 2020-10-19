#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    # The input data are in dataset.data, targets are in dataset.target.

    # If you want to learn about the dataset, uncomment the following line.
    # print(dataset.DESCR)

    # TODO: Append a new feature to all input data, with value "1"
    num_rows, num_column = dataset.data.shape
    ones_column = np.ones((num_rows, 1), dtype='int64')
    featuresWithOnes = np.append(dataset.data, ones_column, axis=1)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    x_train, x_test, y_train, y_test = train_test_split(featuresWithOnes, dataset.target, test_size=args.test_size,
                                                        random_state=args.seed)

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    x_train_transpose = x_train.transpose()
    xTx = x_train_transpose@x_train
    # pprint(xTx)
    inverse = np.linalg.inv(xTx)
    # pprint(inverse)
    weights = inverse @ x_train_transpose @ y_train
    #pprint(y_train.shape)

    # TODO: Predict target values on the test set
    # x_test_transpose = x_test.transpose()
    # weights_transposed = weights.transpose()
    #pprint(x_test.shape)
    # pprint(weights_transposed.shape)
    prediction = x_test @ weights
    # pprint(prediction)
    # pprint(prediction.shape)

    # TODO: Compute root mean square error on the test set predictions
    rmse = mean_squared_error(y_test, prediction, multioutput='uniform_average', squared=False)

    return rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
