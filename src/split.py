import os
from sklearn.model_selection import train_test_split
from src.utils import load_jsonl_as_df, save_df_as_jsonl

def split(input_path: str, output_dir: str):
    """Splits a cleaned datatset into train-validation-test. 
    Stratifies by topic. Saves output as JSONL file to specified 
    directory.
    """
    df = load_jsonl_as_df(input_path)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        stratify=df['topic'], 
        random_state=42
        )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        stratify=temp_df['topic'], 
        random_state=42
        )
    
    os.makedirs(output_dir, exist_ok=True)
    save_df_as_jsonl(train_df, os.path.join(output_dir, 'train.jsonl'))
    save_df_as_jsonl(val_df, os.path.join(output_dir, 'validation.jsonl'))
    save_df_as_jsonl(test_df, os.path.join(output_dir, 'test.jsonl'))
    
    print('Data saved to:')
    print(f' - {output_dir}/train.jsonl ({len(train_df)} records).')
    print(f' - {output_dir}/validation.jsonl ({len(val_df)} records).')
    print(f' - {output_dir}/test.jsonl ({len(test_df)} records).')

