import json
import fitz #PyMuPDF
from pathlib import Path
from tqdm import tqdm

metadata_path = Path("data/papers_metadata.jsonl")
pdf_dir = Path('data/raw_pdfs')
output_path = Path('data/processed_text.jsonl')
output_path.parent.mkdir(parents=True, exist_ok=True)

#Build lookup to match metadata(authors, titles, abstracts, topics) with content
metadata_lookup = {}
with open(metadata_path, 'r', encoding='utf-8') as metafile:
    for line in metafile:
        paper = json.loads(line)
        arxiv_id = paper['arxiv_id']
        metadata_lookup[arxiv_id] = {
            'title': paper.get('title', ''),
            'topic': paper.get('topic', ''),
            'abstract': paper.get('abstract', ''),
            'authors': paper.get('authors', []),
        }

extracted = 0
failed = 0

with open(output_path, 'w', encoding='utf-8') as out_file:
    for pdf in tqdm(list(pdf_dir.glob('*.pdf')), desc='Extracting text from PDF'):
        arxiv_id = pdf.stem
        if arxiv_id not in metadata_lookup:
            continue                    #skip if no metadata is in lookup

        try:
            doc = fitz.open(pdf)
            text = '\n'.join([page.get_text() for page in doc]).strip()

            if text:
                record = {
                    'arxiv_id': arxiv_id,
                    'text': text,
                    **metadata_lookup[arxiv_id]
                }
                out_file.write(json.dumps(record) + '\n')
                extracted +=1
            else:
                failed += 1
        except Exception as e:
            print(f'Failed to extract {arxiv_id}: {e}')
            failed += 1

print(f'Extracted text from {extracted} PDFs.')
print(f'Failed to extract text from {failed} PDFs.')