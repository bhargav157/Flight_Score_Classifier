import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
FILES = list(DATA_DIR.glob('*.csv'))

print('Looking for CSV files in', DATA_DIR)
for f in FILES:
    print('\n---', f.name)
    try:
        df = pd.read_csv(f, nrows=5)
        print('Columns:', list(df.columns))
        print('Sample rows:')
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print('Failed to read:', e)
