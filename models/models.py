import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt

from .neural_network import NN


class NNClassifier():
    def __init__(self, args):
        self.model = NN(args)
        self.lr = args.lr
        self.epochs = args.epochs
        self.optim = args.nn_optim
        self.criterion = args.nn_loss

    def fit(self, X_train, y_train, X_val=None, y_val=None, args=None):
        # Init
        if self.optim == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"{self.optim} is not implemeted.")
        
        if self.criterion == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{self.criterion} is not implemeted.")
        
        train_loss_arr = []
        val_loss_arr = []
        train_acc_arr = []
        val_acc_arr = []
        val_loss = -1
        val_acc = -1

        # Training
        for i in range(self.epochs):
            y_hat = self.model(X_train)
            train_loss = criterion(y_hat, y_train)
            train_loss_arr.append(train_loss.item())

            # Validation
            with torch.no_grad():
                train_acc = self.score(X_train, y_train)
                train_acc_arr.append(train_acc)
                if X_val is not None and y_val is not None:
                    val_loss = criterion(self.model(X_val), y_val)
                    val_loss_arr.append(val_loss.item())
                    val_acc = self.score(X_val, y_val)
                    val_acc_arr.append(val_acc)


            print(f'Epoch: {i} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}')
        
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        fig, ax = plt.subplots()
        plt.plot(train_loss_arr, label="Train")
        plt.plot(val_loss_arr, label="Validation")
        ax.set(title="Loss", xlabel="Epoch", ylabel="Loss")
        ax.grid(which="both")
        ax.legend()
        plt.savefig(f"{args.savedir}/loss.png", bbox_inches="tight")

        fig, ax = plt.subplots()
        plt.plot(train_acc_arr, label="Train")
        plt.plot(val_acc_arr, label="Validation")
        ax.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
        ax.grid(which="both")
        ax.legend()
        plt.savefig(f"{args.savedir}/accuracy.png", bbox_inches="tight")

        plt.close("all")


        return self.model

    def score(self, X, y):

        with torch.no_grad():
            output = self.model(X)
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(y.reshape(1, -1).expand_as(pred))
            correct = correct[:1].reshape(-1).float().sum(0, keepdim=True)
            correct = correct.mul(1.0 / len(X))

        return correct.item()
    
    def predict(self, X):
        
        with torch.no_grad():
            output = self.model(X)
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t().reshape(-1)

        return pred.numpy()



def create_rf(args):
    return RandomForestClassifier()

def create_sgd(args):
    return SGDClassifier(max_iter=args.epochs,
                         loss=args.sgd_loss,
                         penalty=args.sgd_penalty)

def create_nn(args):
    return NNClassifier(args)


model_dict = {
    "random_forest" : create_rf,
    "sgd"           : create_sgd,
    "nn"            : create_nn,
}
