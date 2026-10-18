import csv

with open("experiments.csv") as f:
    rows = list(csv.DictReader(f))

# Find best by val loss
best = min(rows, key=lambda r: float(r["val"]))

print("Best run:")
for k, v in best.items():
    print(f"  {k}: {v}")

print("\nTop 5 by val loss:")
for r in sorted(rows, key=lambda r: float(r["val"]))[:5]:
    print(
        f"  val={r['val']}  config: block={r['block']} emb={r['emb']} hidden={r['hidden']} lr={r['lr']}"
    )
