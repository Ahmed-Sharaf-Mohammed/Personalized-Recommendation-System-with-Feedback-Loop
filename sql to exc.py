import sqlite3
import pandas as pd
import re

def clean_illegal_chars(value):
    if isinstance(value, str):
        return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', value)
    return value
db_file = "db.sqlite3"
excel_file = "output.xlsx"
conn = sqlite3.connect(db_file)

tables_query = """
SELECT name
FROM sqlite_master
WHERE type='table'
AND name NOT LIKE 'sqlite_%';
"""

tables = pd.read_sql_query(tables_query, conn)

with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:

    for table_name in tables['name']:

        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        df = df.map(clean_illegal_chars)

        df.to_excel(writer, sheet_name=table_name[:31], index=False)

        print(f"تم تصدير الجدول: {table_name}")

conn.close()

print(f"\nتم إنشاء الملف: {excel_file}")