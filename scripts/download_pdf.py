import os 
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm

metadata_path = Path("data/papers_metadata.jsonl")
pdf_dir = Path('data/raw_pdfs')
pdf_dir.mkdir(parents=True, exist_ok=True)

with open(metadata_path, 'r', encoding='utf-8') as f:
    papers = [json.loads(line) for line in f]

downloaded = 0
skipped = 0
failed = 0

for paper in tqdm(papers, desc="Downloading PDFs"):
    arxiv_id = paper['arxiv_id']
    pdf_url = paper['pdf_url']
    if not pdf_url:
        skipped += 1
        continue
    
    pdf_path = pdf_dir / f'{arxiv_id}.pdf'
    if pdf_path.exists():
        skipped += 1
        continue                          #Skip already downloaded files

    try:
        response = requests.get(pdf_url, timeout=20)
        response.raise_for_status()
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        downloaded += 1
        time.sleep(1.5)
    except Exception as e:
        print('Failed to downlaod {arxiv_id}: {e}')
        failed += 1

print(f'Done: {downloaded} downloaded, {skipped} skipped, {failed} failed')

    

