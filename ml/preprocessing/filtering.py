import pandas as pd


def iterative_filter(df, min_user_inter=3, min_item_inter=3, max_iter=5):
    for i in range(max_iter):
        print(f"\n🔁 Iteration {i+1}")

        # filter users
        user_counts = df.groupby("user_id").size()
        valid_users = user_counts[user_counts >= min_user_inter].index
        df = df[df["user_id"].isin(valid_users)]

        # filter items
        item_counts = df.groupby("item_id").size()
        valid_items = item_counts[item_counts >= min_item_inter].index
        df = df[df["item_id"].isin(valid_items)]

        print(f"Remaining interactions: {len(df)}")

    return df