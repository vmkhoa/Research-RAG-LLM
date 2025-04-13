import os
import pandas as pd
from src.preprocess import clean_text, chunk_sentences
from src.split import split
from src.utils import load_jsonl_as_df, save_df_as_jsonl
from tqdm import tqdm
tqdm.pandas()

def preprocess(input_path: str, output_path: str):
    """
    Loads JSONL file as dataframe, cleans, tokenizes, and chunks, then 
    saves as a JSONL file.
    """
    df = load_jsonl_as_df(input_path)

    print('Cleaning text ...')
    df['cleaned text'] = df['text'].progress_apply(clean_text)

    print('Chunking text ...')
    df['chunks'] = df['cleaned text'].progress_apply(lambda x: chunk_sentences(x, chunk_size=5))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_df_as_jsonl(df, output_path)
    print(f'Preprocessed data saved to {output_path}')


def main():
    #Configuration
    INPUT_PATH = 'data/extracted_text.jsonl'
    CLEANED_OUTPUT_PATH = 'data/processed/cleaned.jsonl'
    OUTPUT_DIR = 'data'

    #Pipeline
    preprocess(INPUT_PATH, CLEANED_OUTPUT_PATH)
    split(CLEANED_OUTPUT_PATH, OUTPUT_DIR)


if __name__ == "__main__":
    main()
