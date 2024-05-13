import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import SplineTransformer


def sin_cos_transform(col, max_val):
    return np.sin(2 * np.pi * col/max_val), np.cos(2 * np.pi * col/max_val)


def periodic_spline_transformer(df, feature, period, n_splines=None, degree=3, test=False):
    df = df.values.reshape(-1, 1)
    if test:
        spline_transformer = joblib.load(f"encoder/spline_{feature}.gz")
    else:
        if n_splines is None:
            n_splines = period
        n_knots = n_splines + 1  # periodic and include_bias is True
        spline_transformer = SplineTransformer(
            degree=degree,
            n_knots=n_knots,
            knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
            extrapolation="periodic",
            include_bias=True,
        ).fit(df)
        joblib.dump(spline_transformer, f"encoder/spline_{feature}.gz")
    
    splines = spline_transformer.transform(df)
    splines_df = pd.DataFrame(
        splines,
        columns=[f"{feature}_spline_{i}" for i in range(splines.shape[1])],
        )
    return splines_df


def time_encoding(df, args, test=False):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if args.time_encod == "sincos":
        df["hour_of_day_sin"], df["hour_of_day_cos"] = sin_cos_transform(df["timestamp"].dt.hour, 24)
        df["day_of_month_sin"], df["day_of_month_cos"] = sin_cos_transform(df["timestamp"].dt.day, 31)
        df["month_of_year_sin"], df["month_of_year_cos"] = sin_cos_transform(df["timestamp"].dt.month, 12)
    elif args.time_encod == "spline":
        hours_df = periodic_spline_transformer(df["timestamp"].dt.hour, feature="hour", period=24, n_splines=args.num_splines, test=test)
        days_df = periodic_spline_transformer(df["timestamp"].dt.day, feature="day", period=31, n_splines=args.num_splines, test=test)
        months_df = periodic_spline_transformer(df["timestamp"].dt.month, feature="month", period=12, n_splines=args.num_splines, test=test)
        df = pd.concat([df, hours_df, days_df, months_df], axis="columns")
    else:
        raise NotImplementedError(f"{args.time_encod} is not implemeted.")
    
    df["year"] = df["timestamp"].dt.year

    return df.drop(["timestamp"], axis=1)
