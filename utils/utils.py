import json
import pickle
import torch
import pandas as pd

def save_model(clf, args):
    if args.model == "nn":
        torch.save(clf.state_dict(), f"{args.savedir}/{args.model}.pth")
    else:
        pickle.dump(clf, open(f"{args.savedir}/{args.model}.pkl", "wb"))

    with open(f"{args.savedir}/cfg.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=4)

def user_voting(preds, df):
    df = pd.concat([df, pd.DataFrame({"gender_pred": preds})], axis="columns")
    voting_df = df.groupby("user_id")["gender_pred"].mean().round().astype(int)
    user_dict = voting_df.to_dict()
    df["gender_pred"] = df["user_id"].map(user_dict)
    preds = df["gender_pred"].values

    return preds