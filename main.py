import os
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    PrecisionRecallDisplay
)
from preprocess.preprocess import preprocess_data
from models.models import model_dict
from utils.utils import save_model, user_voting



### Config
def get_args_parser():
    parser = argparse.ArgumentParser('Gender Classifier', add_help=False)

    # Save directories
    parser.add_argument('--savedir', default="output", type=str, help="Save directory of trained models")

    # General train parameters
    parser.add_argument('--label-dict', default={"m": 0, "f": 1}, type=dict, help="Label map")
    parser.add_argument('--model', default="sgd", type=str, help="Model Architecture",
                        choices=["random_forest", "sgd", "nn"])
    parser.add_argument('--final-test', default=False, action="store_true", help="Final training?")
    parser.add_argument('--num-classes', default=2, type=int, help="Number of target classes")
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs")
    parser.add_argument('--lr', default=0.001, type=float, help="Initial Learning Rate")
    parser.add_argument('--norm', default=True, action="store_false", help="Apply normalization?")
    parser.add_argument('--cat-encod', default="target", type=str, help="Encoding of categorical features",
                        choices=["target", "one_hot", "freq", "bin"])
    parser.add_argument('--time-encod', default="sincos", type=str, help="Encoding of time features",
                        choices=["sincos", "spline"])
    parser.add_argument('--rbf', default=False, action="store_true", help="Radial Basis Function feature transformation")
    parser.add_argument('--user-voting', default=True, action="store_false", help="Apply user voting?")


    # Target parameters
    parser.add_argument('--mean-weight', default=10, type=int, help="Weight of smoothed mean")

    # Spline parameters
    parser.add_argument('--num-splines', default=12, type=int, help="Number of splines for SplineTransformer")

    # Random Forest parameters


    # SGDClassifier parameters
    parser.add_argument('--sgd_loss', default="hinge", type=str, help="Loss function of SGD classifier",
                        choices=["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron", "squared_error",
                                 "huber", "epsilon_insensitive", "squared_epsilon_insensitive"])
    parser.add_argument('--sgd_penalty', default="l2", type=str, help="Penalty type of SGD classifier",
                        choices=["l2", "l1", "elasticnet"])

    # Neural Network parameters
    parser.add_argument('--nn-optim', default="adam", type=str, help="Optimizer of Neural Network")
    parser.add_argument('--nn-loss', default="cross_entropy", type=str, help="Criterion of Neural Network")

    return parser


def main(args):
    args.savedir = os.path.join(args.savedir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.savedir)

    ### Load data ###
    print("Loading data...")
    train_df, val_df, test_df = pd.read_csv("data/train.csv"), pd.read_csv("data/val.csv"), pd.read_csv("data/test.csv")

    if args.final_test:
        train_df = pd.concat([train_df, val_df])
        val_df = test_df


    ### Preprocess data ###
    print("Preprocessing data...")
    X_train, y_train = preprocess_data(train_df, args, test=False)
    X_val, y_val = preprocess_data(val_df, args, test=True)

    args.in_features = X_train.shape[-1]


    ### Initialize model ###
    clf = model_dict[args.model](args)


    ### Fit model ###
    print("Training model...")
    if args.model == "nn":
        trained_clf = clf.fit(X_train, y_train, X_val, y_val, args)
    else:
        trained_clf = clf.fit(X_train, y_train)


    ### Save model and config ###
    print("Saving model...")
    save_model(trained_clf, args)

    ### Evaluate model ###
    print("Evaluating model...")
    train_score = clf.score(X_train, y_train)
    preds = clf.predict(X_val)
    if args.user_voting:
        preds = user_voting(preds, val_df)

    if args.final_test:
        preds = np.vectorize({v: k for k, v in args.label_dict.items()}.__getitem__)(preds)
        val_df = pd.concat([test_df, pd.DataFrame({"gender": preds})], axis="columns")
        val_df.to_csv("data/test_pred.csv", index=False)
    else:
        val_score = accuracy_score(y_val, preds)
        print(f"Train score: {train_score:.2f}")
        print(f"Validation score: {val_score:.2f}")

        display = PrecisionRecallDisplay.from_predictions(y_val, preds, name=args.model)
        _ = display.ax_.set_title("Gender Classification Precision Recall")
        plt.grid(which="both")
        plt.savefig(f"{args.savedir}/precision_recall.png", bbox_inches="tight")

        plt.close("all")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gender Classifier', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
