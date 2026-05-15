import numpy as np
import pandas as pd
import os

df = pd.read_csv('Max_AAV_df.csv')
print(f"Shape: {df.shape}")

# Split into 4 chunks
chunk_size = len(df) // 4
for i in range(4):
    start = i * chunk_size
    end = start + chunk_size if i < 3 else len(df)  # last chunk gets remainder
    chunk = df.iloc[start:end]
    chunk.to_csv(f'Max_AAV_df_part{i}.csv', index=False)
    size = os.path.getsize(f'Max_AAV_df_part{i}.csv') / (1024*1024)
    print(f"Max_AAV_df_part{i}.csv: {size:.2f} MB (rows {start}-{end})")
