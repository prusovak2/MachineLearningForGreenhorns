#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from pprint import pprint

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.9, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    # TODO: Split the dataset into a train set and a test set.
    features = dataset.data
    target = dataset.target
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=args.test_size,
                                                        random_state=args.seed)

    lambdas = np.geomspace(0.01, 100, num=500)
    # TODO: Using `sklearn.linear_model.Ridge`, fit the train set using
    # L2 regularization, employing above defined lambdas.
    # For every model, compute the root mean squared error
    # (do not forget `sklearn.metrics.mean_squared_error`) and return the
    # lambda producing lowest test error.

    so_far_best_lambda = None
    so_far_best_rmse = 4200000
    rmses = []
    for lam in lambdas:
        model = sklearn.linear_model.Ridge(alpha=lam)
        model.fit(x_train, y_train)
        y_predicted = model.predict(x_test)
        curr_rmse = sklearn.metrics.mean_squared_error(y_test, y_predicted, squared=False)
        rmses.append(curr_rmse)
        if curr_rmse <= so_far_best_rmse:
            so_far_best_lambda = lam
            so_far_best_rmse = curr_rmse

    best_lambda = so_far_best_lambda
    best_rmse = so_far_best_rmse

    if args.plot:
        # This block is not required to pass in ReCodEx, however, it is useful
        # to learn to visualize the results.

        # If you collect the respective results for `lambdas` to an array called `rmse`,
        # the following lines will plot the result if you add `--plot` argument.
        import matplotlib.pyplot as plt
        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength")
        plt.ylabel("RMSE")
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))
