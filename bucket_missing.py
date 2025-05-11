from collections import Counter
import json

# Load missing tickers
with open('missing_pairs.json') as f:
    missing = list(json.load(f).keys())

# Define simple heuristics for common “exotic” suffixes
def classify(ticker):
    t = ticker.upper()
    if t.endswith(('.U', 'U')):
        return 'Unit'
    if t.endswith(('.WS', 'W')):
        return 'Warrant'
    if t.endswith(('.R', 'R')):
        return 'Right'
    if t.endswith(('.P', 'P')):
        return 'Preferred'
    # you can expand with other patterns (e.g. ‘-PD’, etc.)
    return 'Common/Other'

# Count them
counts = Counter(classify(t) for t in missing)
print("Missing ticker counts by class:")
for cls, cnt in counts.most_common():
    print(f"  {cls:12s}: {cnt}")

# List a few examples in each bucket
examples = {}
for t in missing:
    cls = classify(t)
    examples.setdefault(cls, []).append(t)
for cls, ex in examples.items():
    print(f"\n{cls} examples: {ex[:5]}")
