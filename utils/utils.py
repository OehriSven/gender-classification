import json
import pickle
import torch

def save_model(clf, args):
    if args.model == "nn":
        torch.save(clf.state_dict(), f"{args.savedir}/{args.model}.pth")
    else:
        pickle.dump(clf, open(f"{args.savedir}/{args.model}.pkl", "wb"))

    with open(f"{args.savedir}/cfg.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=4)