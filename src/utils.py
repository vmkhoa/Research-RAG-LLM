import pandas as pd

def load_jsonl_as_df(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)

def save_df_as_jsonl(df: pd.DataFrame, path: str):
    df.to_json(path, orient='records', lines=True)