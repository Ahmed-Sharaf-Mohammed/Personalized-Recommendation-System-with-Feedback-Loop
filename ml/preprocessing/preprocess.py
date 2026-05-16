import gzip
import json
import pandas as pd
import os
from ml.utils.mapper import DataMapper
from ml.preprocessing.eda import run_eda
from ml.preprocessing.filtering import iterative_filter


# الخطوات الرئيسية في هذا الكود تشمل:
# 1. تحميل البيانات من الملفات المضغوطة.
# 2. تحويل البيانات إلى تنسيق موحد باستخدام DataMapper.
# 3. حفظ البيانات المعالجة في ملفات Parquet.
# 4. تشغيل تحليل استكشافي للبيانات (EDA) للحصول على إحصائيات أولية عن البيانات.



def run_preprocessing(
    interactions_file,
    meta_file,
    output_dir="data/processed/",
    max_rows=50000
):
    mapper = DataMapper()

    interactions = []
    item_ids = set()

    print("🔄 Loading interactions...")

    with gzip.open(interactions_file, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):  # data will be like : {"reviewerID": "A2SUAM1J3GNN3B", "asin": "0000013714", "reviewerName": "J. McDonald", "helpful": [2, 3], "reviewText": "I bought this for my husband who plays the piano. He has been having a wonderful time playing these old songs on his piano. The music is very well written and easy to read.", "overall": 5.0, "summary": "Heaven in the form of music", "unixReviewTime": 1303862400, "reviewTime": "05 26, 2011"}
            if i >= max_rows:
                break

            row = json.loads(line)
            mapped = mapper.map_interaction(row)

            if mapped["user_id"] and mapped["item_id"]:
                interactions.append(mapped)
                item_ids.add(mapped["item_id"])

    print(f"✅ Interactions loaded: {len(interactions)}")

    print("🔄 Loading items...")

    items = []
    with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)

            if row.get("parent_asin") in item_ids:
                mapped = mapper.map_item(row)
                items.append(mapped)

    print(f"✅ Items loaded: {len(items)}")

    # DataFrames
    df_inter = pd.DataFrame(interactions)
    df_items = pd.DataFrame(items)

    # basic cleaning
    df_inter = df_inter.dropna(subset=["user_id", "item_id"])
    df_items = df_items.drop_duplicates(subset=["item_id"])


    print("🧹 Applying filtering...")
    df_inter = iterative_filter(df_inter)


    # filter items based on remaining interactions
    valid_item_ids = df_inter["item_id"].unique()
    df_items = df_items[df_items["item_id"].isin(valid_item_ids)]


    # save
    os.makedirs(output_dir, exist_ok=True)

    df_inter.to_parquet(f"{output_dir}/interactions.parquet", index=False)
    df_items.to_parquet(f"{output_dir}/items.parquet", index=False)

    print("💾 Data saved successfully")



run_preprocessing("data/raw/Electronics.jsonl.gz", "data/raw/meta_Electronics.jsonl.gz")
run_eda()