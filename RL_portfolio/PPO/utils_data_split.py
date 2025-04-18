def train_test_split_df(df, train_ratio=0.8):
    dates = df.index.get_level_values("date").unique().sort_values()
    split_point = int(len(dates) * train_ratio)
    train_dates = dates[:split_point]
    test_dates = dates[split_point:]
    df_train = df[df.index.get_level_values("date").isin(train_dates)]
    df_test = df[df.index.get_level_values("date").isin(test_dates)]
    return df_train, df_test