import arxiv
import json
from pathlib import Path

output_path = Path('data/')
output_path.mkdir(parents=True, exist_ok=True)

topics = {
    "artificial intelligence": 85,
    "computer science": 85,
    "mathematics": 85,
    "physics": 85,
    "biology": 80,
    "social sciences": 80
}

all_papers = []

#Access arxiv API Python wrapper from https://github.com/lukasschwab/arxiv.py
client = arxiv.Client()
for topic, count in topics.items():
    search = arxiv.Search(
        query=topic,
        max_results=count,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    for result in client.results(search):
        paper = {
            'title': result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary,
            "pdf_url": result.pdf_url,
            "topic": topic,
            "published": result.published.isoformat(),
            "arxiv_id": result.entry_id.split("/")[-1]
        }
        all_papers.append(paper)
#Save to JSONL
with open(output_path/'papers_metadata.jsonl', 'w', encoding='utf-8') as f:
    for paper in all_papers:
        f.write(json.dumps(paper) + '\n')

print(f"Collected {len(all_papers)} papers total.")



