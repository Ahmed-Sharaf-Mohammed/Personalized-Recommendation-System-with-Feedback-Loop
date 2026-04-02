import pandas as pd


def run_eda(processed_dir="data/processed/"):
    print("📊 Loading data...")
    df_inter = pd.read_parquet(f"{processed_dir}/interactions.parquet")
    df_items = pd.read_parquet(f"{processed_dir}/items.parquet")

    print("\n🔢 Basic Stats:")
    print(f"Interactions: {len(df_inter)}")
    print(f"Unique Users: {df_inter['user_id'].nunique()}")
    print(f"Unique Items: {df_inter['item_id'].nunique()}")

    print("\n📈 Interactions per user:")
    user_counts = df_inter.groupby("user_id").size()
    print(user_counts.describe())

    print("\n📈 Interactions per item:")
    item_counts = df_inter.groupby("item_id").size()
    print(item_counts.describe())

    print("\n⭐ Ratings distribution:")
    print(df_inter["rating"].value_counts().sort_index())

    print("\n⚠️ Cold Users (<=2 interactions):")
    cold_users = (user_counts <= 2).sum()
    print(f"{cold_users} users")

    print("\n⚠️ Cold Items (<=2 interactions):")
    cold_items = (item_counts <= 2).sum()
    print(f"{cold_items} items")
