# Load user words
with open("words.txt", "r") as f:
    words = [w.strip().lower() for w in f.readlines()]

# Load QuickDraw categories
with open("categories.txt", "r") as f:
    categories = [c.strip().lower() for c in f.readlines()]

# Compare
matches = sorted(set(words) & set(categories))

print(f"Total words: {len(words)}")
print(f"Matches: {len(matches)}\n")

for m in matches:
    print(m)
