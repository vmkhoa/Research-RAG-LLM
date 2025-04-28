import json

input_path = 'data/papers_metadata.jsonl'
output_path = 'topics_list.txt'

titles = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        paper = json.loads(line)
        title = paper.get('title')
        if title:
            titles.append(title)
        
with open(output_path, 'w', encoding='utf-8') as f:
    for title in titles:
        f.write(title.strip() + '\n')

print(f"Extracted {len(titles)} titles to {output_path}")