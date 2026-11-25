import matplotlib.pyplot as plt
from openpyxl import load_workbook

FILE = "Horizontal_Format.xlsx"
MIN_SUP = 0.4
MIN_CONF = 0.6

def load_transactions(path):
    rows = load_workbook(path).active.iter_rows(min_row=2, values_only=True) 
    transactions = []
    for row in rows:
        if row[1]:
            transactions.append({item.strip() for item in str(row[1]).split(",")})
    return transactions

def support(itemset, transactions):
    count = 0
    for t in transactions:
        if itemset <= t:
            count += 1
    return count / len(transactions)

def keep_frequent(candidates, transactions, min_sup):
    frequent = {}
    for c in candidates:
        s = support(c, transactions)
        if s >= min_sup - 1e-9:
            frequent[c] = s
    return frequent

def make_candidates(prev_level, k):
    candidates = set()
    for a in prev_level:
        for b in prev_level:
            union = a | b
            if len(union) == k:
                if all(union - {item} in prev_level for item in union):
                    candidates.add(union)
    return candidates

def apriori(transactions, min_sup):
    items = set()
    for t in transactions:
        items |= t

    level = keep_frequent([frozenset([i]) for i in items], transactions, min_sup)
    levels = []
    k = 1
    while level:
        levels.append(level)
        k += 1
        level = keep_frequent(make_candidates(level, k), transactions, min_sup)
    return levels

def splits(itemset):
    items = sorted(itemset)
    for mask in range(1, 2 ** len(items) - 1):
        left = {items[i] for i in range(len(items)) if mask >> i & 1}
        yield frozenset(left), itemset - left

def get_rules(levels, min_conf):
    sup = {}
    for level in levels:
        sup.update(level)
    rules = []
    for itemset, s in sup.items():
        for left, right in splits(itemset):
            conf = s / sup[left]
            lift = conf / sup[right]
            rules.append((left, right, s, conf, lift))
    strong = [r for r in rules if r[3] >= min_conf - 1e-9]
    return rules, strong


def show(itemset):
    return "{" + ", ".join(sorted(itemset)) + "}"


def lift_text(lift):
    if abs(lift - 1) < 1e-9:
        return "independent"
    return "positive" if lift > 1 else "negative"

def plot_levels(levels, min_sup):
    for k, level in enumerate(levels, 1):
        items = sorted(level.items(), key=lambda x: (-x[1], sorted(x[0])))
        labels = [", ".join(sorted(itemset)) for itemset, _ in items]
        values = [s for _, s in items]

        plt.figure(figsize=(max(6, len(labels) * 0.7), 4))
        bars = plt.bar(labels, values, color="steelblue")
        plt.bar_label(bars, fmt="%.2f", fontsize=8)
        plt.axhline(min_sup, color="red", linestyle="--", label=f"min support = {min_sup}")
        plt.ylim(0, 1.05)
        plt.title(f"L{k} frequent itemsets - support")
        plt.xlabel("Itemset")
        plt.ylabel("Support")
        plt.xticks(rotation=45, ha="right")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"L{k}_support.png", dpi=120)
    plt.show()

def main():
    transactions = load_transactions(FILE)
    print(f"Loaded {len(transactions)} transactions")

    levels = apriori(transactions, MIN_SUP)

    for k, level in enumerate(levels, 1):
        print(f"\nL{k}: {len(level)} frequent {k}-itemsets")
        for itemset, s in sorted(level.items(), key=lambda x: (-x[1], sorted(x[0]))):
            print(f"  {show(itemset)}  support = {s:.2f}")

    _, strong = get_rules(levels, MIN_CONF)
    strong.sort(key=lambda r: (r[3], r[4], r[2]), reverse=True)
    
    print(f"\nTop 20 strong rules (confidence >= {MIN_CONF}):")
    for left, right, s, conf, lift in strong[:20]:
        print(f"  {show(left)} -> {show(right)}  "
              f"sup={s:.2f} conf={conf:.2f} lift={lift:.2f} ({lift_text(lift)})")
        
    print("\nDifferent minimum support / confidence values")
    print(f"{'min_sup':>8} {'min_conf':>9} {'max k':>6} {'#itemsets':>10} {'#rules':>7} {'#strong':>8}")
    for ms in (0.2, 0.4, 0.6, 0.8):
        lv = apriori(transactions, ms)
        for mc in (0.5, 0.7, 0.9):
            all_rules, strong_rules = get_rules(lv, mc)
            total = sum(len(level) for level in lv)
            print(f"{ms:>8} {mc:>9} {len(lv):>6} {total:>10} {len(all_rules):>7} {len(strong_rules):>8}")

    plot_levels(levels, MIN_SUP)

if __name__ == "__main__":
    main()