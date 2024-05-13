import joblib
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

from .cat_transform import cat_encoding
from .time_transform import time_encoding


def preprocess_data(df, args, test=False):
    # Label transformation
    df = df.replace({"gender": args.label_dict})

    # Categorical Encoding
    df = cat_encoding(df, args, test=test)

    # Time Encoding
    df = time_encoding(df, args, test=test)

    if "gender" in df.columns:
        X, y = df.drop(["gender"], axis=1).values, df["gender"].values
    else:
        X, y = df.values, []

    # Standardization
    if args.norm:
        if not test:
            scaler = StandardScaler().fit(X)
            joblib.dump(scaler, "encoder/scaler.gz")
        else:
            scaler = joblib.load("encoder/scaler.gz")
        X = scaler.transform(X)

    ### RBF Transformation
    if args.rbf:
        if not test:
            rbf_feature = RBFSampler().fit(X)
            joblib.dump(rbf_feature, "encoder/rbf_feature.gz")
        else:
            rbf_feature = joblib.load("encoder/rbf_feature.gz")
        X = rbf_feature.transform(X)

    # Tensor Transformation
    if args.model == "nn":
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

    return X, y
