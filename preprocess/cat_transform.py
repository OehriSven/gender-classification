import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def one_hot_transform(df, features, test):
    if test:
        transformer = joblib.load("encoder/one_hot.gz")
        return pd.DataFrame(transformer.transform(df))

    transformer = ColumnTransformer([("one_hot",
                                    OneHotEncoder(sparse_output=False),
                                    features)],
                                    remainder="passthrough")
    transformer = transformer.fit(df)
    joblib.dump(transformer, "encoder/one_hot.gz")

    return pd.DataFrame(transformer.transform(df))


def freq_transform(df, by, test):
    if test:
        counts = pd.read_csv("encoder/freq_encoding.csv", index_col=0).squeeze()
    else:
        counts = df[by].value_counts()
        counts.to_csv("encoder/freq_encoding.csv")
    
    df[by] = df[by].map(counts)

    return df


def bin_transform(df, by, encoding, test):
    if test:
        path_index = json.load(open("encoder/bin_encoding.json"))
    else:
        path_index = dict(zip(df[by].unique(), range(len(df[by].unique()))))
        json.dump(path_index, open("encoder/bin_encoding.json", 'w'))

    # Label encoding
    df[by] = df[by].map(path_index)

    # Binary encoding
    if encoding == "bin":
        binary_rep = df[by].apply(lambda x: bin(x)[2:].zfill(len(bin(len(df[by].unique())))))

        for i in range(len(binary_rep.iloc[0])):
            df[f'bit_{i}'] = binary_rep.apply(lambda x: int(x[i]))
        
        df = df.drop([by], axis=1)
    return df


def target_transform(df, by, on, m, test):
    if test:
        target_enc = pd.read_csv("encoder/target_encoding.csv", index_col=0).squeeze()
        return df[by].map(target_enc)

    # Global mean
    mean = df[on].mean()

    # Number of values and mean of each group
    agg = df.groupby(by)[on].agg(["count", "mean"])
    counts = agg["count"]
    means = agg["mean"]

    # Smoothed mean
    target_enc = (counts * means + m * mean) / (counts + m)

    target_enc.to_csv("encoder/target_encoding.csv")

    # Replace features with smoothed mean
    return df[by].map(target_enc)


def cat_encoding(df, args, user_id=False, test=False):
    if not user_id:
        df = df.drop(["user_id"], axis=1)

    if args.cat_encod == "target":
        df["path"] = target_transform(df, by="path", on="gender", m=args.mean_weight, test=test)

    elif args.cat_encod == "one_hot":
        df = one_hot_transform(df, features=["path"], test=test)
    
    elif args.cat_encod == "freq":
        df = freq_transform(df, by="path", test=test)

    elif args.cat_encod == "bin" or args.cat_encod == "label":
        df = bin_transform(df, by="path", encoding=args.cat_encod, test=test)
    
    else:
        raise NotImplementedError(f"{args.cat_encod} is not implemeted.")

    return df
