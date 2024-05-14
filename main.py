import pickle
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, PrecisionRecallDisplay

from preprocess.preprocess import preprocess_data
from models.models import model_dict



### Config
def get_args_parser():
    parser = argparse.ArgumentParser('Gender Classifier', add_help=False)

    # General train parameters
    parser.add_argument('--label-dict', default={"m": 0, "f": 1}, type=dict, help="Label map")
    parser.add_argument('--model', default="sgd", type=str, help="Model Architecture",
                        choices=["random_forest", "sgd", "nn"])
    parser.add_argument('--final-test', default=False, action="store_true", help="Final training?")
    parser.add_argument('--num_classes', default=2, type=int, help="Number of target classes")
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs")
    parser.add_argument('--lr', default=0.001, type=float, help="Initial Learning Rate")
    parser.add_argument('--norm', default=True, action="store_false", help="Apply normalization?")
    parser.add_argument('--cat-encod', default="target", type=str, help="Encoding of categorical features",
                        choices=["target", "one_hot", "freq", "bin"])
    parser.add_argument('--time-encod', default="sincos", type=str, help="Encoding of time features",
                        choices=["sincos", "spline"])
    parser.add_argument('--rbf', default=False, action="store_true", help="Radial Basis Function feature transformation")

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
    parser.add_argument('--nn_optim', default="adam", type=str, help="Optimizer of Neural Network")
    parser.add_argument('--nn_loss', default="cross_entropy", type=str, help="Criterion of Neural Network")

    return parser


def main(args):
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
    trained_clf = clf.fit(X_train, y_train)


    ## Save model ###
    print("Saving model...")
    if args.model == "nn":
        torch.save(trained_clf.state_dict(), f"pytorch_models/{args.model}_{args.cat_encod}_{args.time_encod}_{args.epochs}")
    else:
        pickle.dump(trained_clf, open(f"sklearn_models/{args.model}_{args.cat_encod}_{args.time_encod}_{args.epochs}.pkl", "wb"))


    ### Evaluate model ###
    print("Evaluating model...")
    train_score = clf.score(X_train, y_train)
    print(f"Train score: {train_score:.2f}")

    if args.final_test:
        preds = clf.predict(X_val)
        preds = np.vectorize({v: k for k, v in args.label_dict.items()}.__getitem__)(preds)
        test_df = pd.concat([test_df, pd.DataFrame({"gender": preds})], axis="columns")
        test_df.to_csv("data/test_pred.csv", index=False)
    else:
        val_score = clf.score(X_val, y_val)
        conf_matrix = confusion_matrix(y_val, clf.predict(X_val))
        print(f"Validation score: {val_score:.2f}")
        print(f"Confusion Matrix: {conf_matrix}")
        display = PrecisionRecallDisplay.from_estimator(clf, X_val, y_val, name=args.model)
        _ = display.ax_.set_title("Gender Classification Precision Recall curve")
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gender Classifier', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
