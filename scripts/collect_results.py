"""Collect ``results/logs/test_*.txt`` into a Markdown table (``results/tables/collected.md``)."""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERN = re.compile(r"^(?P<dataset>\S+) (?P<way>\d+)-way (?P<shot>\d+)-shot: acc = (?P<acc>[\d.]+) \+- (?P<ci>[\d.]+)")


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "logs", "test_*.txt"))):
        with open(path) as f:
            for line in f:
                m = PATTERN.match(line.strip())
                if m:
                    rows.append((m["dataset"], int(m["way"]), int(m["shot"]), float(m["acc"]), float(m["ci"]),
                                 os.path.basename(path)))
    out = os.path.join(ROOT, "results", "tables", "collected.md")
    with open(out, "w") as f:
        f.write("| Dataset | Setting | Accuracy (%) | Log |\n|:--|:--:|:--:|:--|\n")
        for d, w, s, acc, ci, log in rows:
            f.write(f"| {d} | {w}-way {s}-shot | {acc:.2f} ± {ci:.2f} | {log} |\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
