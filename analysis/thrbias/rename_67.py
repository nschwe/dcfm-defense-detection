#!/usr/bin/env python3
"""rename_67.py -- the 67 engineered columns, under the names the paper uses.

WHAT IT DOES
    Reads the generated feature table, applies the canonical -> paper mapping of
    paper_names.py, and prints the 67 grouped by family with a short definition
    of each. Nothing is written unless --out is given.

WHY A RENAME IS NEEDED AT ALL
    arm_spec.py feeds every arm the same canonical column names so the pipeline
    is identical across arms. Twelve of the seventeen listener observables are
    therefore reported under the name of the network-wide metric they replace,
    and the quantity is not the same one. The *Flow* family is the worst case:
    those six are per-SOURCE quantities computed from frames the listener heard,
    not per-flow quantities reconstructed end to end.

THE ONE EXCEPTION
    paper_names.py maps AverageHopCount -> MeanTcHopCount. For the appendix table
    AverageHopCount is intentionally retained under its canonical name, and is
    given its own definition there instead. The override below applies that.

USAGE
    python3 rename_67.py                  # print the 67, grouped by family
    python3 rename_67.py --tex FILE.tex   # the appendix table, as a longtable
    python3 rename_67.py --out FILE.csv   # the same rows, as CSV

    --table and --names override the two inputs, which otherwise resolve
    against this file: out67/supplementary_features.csv and ../paper_names.py.
"""
import argparse
import csv
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Resolved against this file, so a clone of the repository runs as it stands.
# In the tree these are analysis/thrbias/out67/supplementary_features.csv
# and analysis/paper_names.py, which is what the absolute paths named.
_HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(_HERE, "out67", "supplementary_features.csv")
NAMES = os.path.join(os.path.dirname(_HERE), "paper_names.py")

# AverageHopCount is intentionally retained under its canonical name
# for the appendix table, so the general naming rule is not applied here.
OVERRIDE = {"AverageHopCount": "AverageHopCount"}

FAMILY = [
    ("base", "the seventeen single-vantage observables"),
    ("network-performance indicator", "products and sums of the observables"),
    ("statistical: log1p", "log1p of each observable"),
    ("statistical: sqrt", "square root of each observable"),
    ("statistical: power", "powers of the observed data rate"),
    ("statistical: row-wise", "statistics across the observables of one window"),
]


def load_mapping(path):
    """canonical -> paper name. None means a duplicate that is not reported."""
    src = open(path, encoding="utf-8").read()
    blk = src[src.index("PAPER_NAMES = {"):src.index("\nCATEGORY")]
    out = {}
    for m in re.finditer(r'"([A-Za-z_0-9]+)":\s*\(\s*\n?\s*("([^"]*)"|None)', blk):
        out[m.group(1)] = m.group(3)
    out.update(OVERRIDE)
    return out


def load_defs(path):
    """canonical -> the definition paper_names.py carries.

    The definitions are written as ADJACENT string literals across several
    lines, which Python concatenates. A regex that captures one literal
    truncates the sentence mid-word, so every run of adjacent literals is
    joined here.
    """
    src = open(path, encoding="utf-8").read()
    blk = src[src.index("PAPER_NAMES = {"):src.index("\nCATEGORY")]
    out = {}
    # entry: "canonical": (<paper name or None>, <definition literals>, <source expr>)
    for m in re.finditer(r'"([A-Za-z_0-9]+)":\s*\(\s*((?:[^()]|\([^()]*\))*?)\)\s*,', blk):
        can, body = m.group(1), m.group(2)
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"|(\bNone\b)', body)
        lits, run, seen_name = [], [], False
        for text, none in parts:
            if none:
                lits.append(None)
                continue
            lits.append(text)
        if not lits:
            continue
        # element 0 is the paper name (or None); the definition is the run of
        # literals that follows it, up to the source expression (the last one)
        rest = lits[1:]
        if len(rest) >= 2:
            rest = rest[:-1]
        defn = " ".join(x for x in rest if x)
        if defn:
            out[can] = " ".join(defn.split())
    return out


def rename(feature, mapping):
    """A derived column carries its base's name, so renaming the base renames it."""
    for prefix in ("log1p_", "sqrt_"):
        if feature.startswith(prefix):
            base = feature[len(prefix):]
            return prefix + mapping.get(base, base)
    for suffix in ("_squared", "_cubed"):
        if feature.endswith(suffix):
            base = feature[:-len(suffix)]
            return mapping.get(base, base) + suffix
    return mapping.get(feature, feature)


def rewrite_formula(formula, mapping):
    """Replace whole-word canonical names inside a formula."""
    def sub(m):
        return mapping.get(m.group(0), m.group(0))
    return re.sub(r"[A-Za-z_][A-Za-z_0-9]*", sub, formula)


# The row-wise family is computed across the seventeen observables of one
# measurement window. The generated table describes all twelve with one generic
# sentence, which says nothing; each is given its own here.
ROWWISE = {
    "row_mean": "mean of the seventeen observables in this window",
    "row_std": "spread of the seventeen, as a standard deviation",
    "row_median": "middle value of the seventeen",
    "row_max": "largest of the seventeen",
    "row_min": "smallest of the seventeen",
    "row_range": "largest minus smallest",
    "row_cv": "spread relative to the mean, row_std / row_mean",
    "row_skew": "how lopsided the seventeen are about their mean",
    "row_kurtosis": "how heavy the tails of the seventeen are",
    "row_q25": "value below which a quarter of the seventeen fall",
    "row_q75": "value below which three quarters fall",
    "row_iqr": "row_q75 minus row_q25, the middle half's width",
}

PAIRWISE = {
    "TDR": "how much topology each second carries: message rate times links per message",
    "Total_Traffic": "every message rate the listener hears, added together",
}


def short(feature, canonical, defs, mapping, formula, description):
    """A few words saying what the column actually is."""
    for prefix, word in (("log1p_", "log1p of"), ("sqrt_", "square root of")):
        if canonical.startswith(prefix):
            base = canonical[len(prefix):]
            return "%s %s" % (word, mapping.get(base, base))
    for suffix, word in (("_squared", "squared"), ("_cubed", "cubed")):
        if canonical.endswith(suffix):
            base = canonical[:-len(suffix)]
            return "%s, %s" % (mapping.get(base, base), word)
    if canonical in ROWWISE:
        return ROWWISE[canonical]
    if canonical in PAIRWISE:
        return PAIRWISE[canonical]
    if canonical in defs:
        return defs[canonical]
    # the generated table's own description is generic for some families; the
    # formula is the specific thing, so prefer it
    return " ".join((formula or description or "").split())


def tex_escape(s):
    """Only what a feature name or a plain definition can contain."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def breakable(name):
    """A 40-character typewriter name will not wrap on its own and pushes the
    table off the page. Offer break points after each underscore and before each
    internal capital, which is where a reader would break the name anyway."""
    BS = chr(92)
    out, prev = [], ""
    for i, ch in enumerate(name):
        if ch == "_":
            out.append(BS + "_" + BS + "allowbreak{}")
        else:
            if i and ch.isupper() and prev.islower():
                out.append(BS + "allowbreak{}")
            out.append(ch)
        prev = ch
    return "".join(out)


def emit_tex(rows, path):
    """A longtable: 67 rows do not fit a table environment on one page.

    Requires \\usepackage{longtable} in the preamble. The caption is wrapped in \\hl, the convention the
    other added tables follow.
    """
    BS = chr(92)
    L = []
    w = L.append
    w("% Generated by rename_67.py. Do not edit by hand: regenerate.")
    w("% Requires " + BS + "usepackage{longtable} and " + BS + "usepackage{array}.")
    w("% Widths: 0.30 + 0.40 of " + BS + "linewidth for the two text columns, the")
    w("% rest for the number and the two ticks. Names carry " + BS + "allowbreak so a")
    w("% long one wraps inside its column instead of overflowing the page.")
    w("{" + BS + "footnotesize")
    w(BS + "setlength{" + BS + "tabcolsep}{4pt}")
    w(BS + "begin{longtable}{@{}r>{" + BS + "ttfamily" + BS + "raggedright"
      + BS + "arraybackslash}p{0.30" + BS + "linewidth}"
      + ">{" + BS + "raggedright" + BS + "arraybackslash}p{0.40" + BS + "linewidth}"
      + "cc@{}}")
    w(BS + "caption{" + BS + "hl{The 67 engineered columns, with the "
      "configuration that retains each. The seventeen single-vantage "
      "observables are listed first; the remaining fifty are derived from "
      "them alone. Names are those used throughout this paper.}}")
    w(BS + "label{tab:feature-list}" + BS + BS)
    w(BS + "toprule")
    w("\\# & Feature & Definition & Static & Mobile " + BS + BS)
    w(BS + "midrule")
    w(BS + "endfirsthead")
    w(BS + "toprule")
    w("\\# & Feature & Definition & Static & Mobile " + BS + BS)
    w(BS + "midrule")
    w(BS + "endhead")
    w(BS + "bottomrule")
    w(BS + "endfoot")

    def tick(v):
        return BS + "checkmark" if str(v).strip().lower() in ("1", "true", "yes", "y") else ""

    n = 0
    for kind, caption in FAMILY:
        grp = [x for x in rows if x["family"] == kind]
        if not grp:
            continue
        # A family header set in italics at the body size is hard to pick out
        # of sixty-seven rows. Bold, with a rule above it and air on both
        # sides, so the eye finds the group boundaries.
        if n:
            w(BS + "addlinespace[6pt]")
            w(BS + "midrule")
        w(BS + "addlinespace[3pt]")
        w(BS + "multicolumn{5}{@{}l@{}}{" + BS + "textbf{"
          + tex_escape(caption[0].upper() + caption[1:]) + "} (" + str(len(grp)) + ")} "
          + BS + BS)
        w(BS + "addlinespace[4pt]")
        for x in grp:
            n += 1
            w("%d & %s & %s & %s & %s %s"
              % (n, breakable(x["feature"]),
                 tex_escape(x["definition"]),
                 tick(x["survives_variance_correlation_filters_static"]),
                 tick(x["survives_variance_correlation_filters_mobile"]), BS + BS))
    w(BS + "end{longtable}")
    w("}")
    open(path, "wb").write(("\r\n".join(L) + "\r\n").encode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--names", default=NAMES)
    ap.add_argument("--out", help="write the renamed table here")
    ap.add_argument("--tex", help="write a LaTeX longtable for the appendix")
    a = ap.parse_args()

    mapping = load_mapping(a.names)
    defs = load_defs(a.names)
    rows = list(csv.DictReader(open(a.table, newline="", encoding="utf-8")))

    renamed = []
    for r in rows:
        new = rename(r["feature"], mapping)
        renamed.append({
            "family": r["kind"],
            "feature": new,
            "canonical": r["feature"],
            "changed": "yes" if new != r["feature"] else "",
            "definition": short(new, r["feature"], defs, mapping,
                                r.get("formula", ""), r.get("description", "")),
            "formula": rewrite_formula(r.get("formula", ""), mapping),
            "constant": r.get("constant", ""),
            "survives_variance_correlation_filters_static": r.get("survives_variance_correlation_filters_static", ""),
            "survives_variance_correlation_filters_mobile": r.get("survives_variance_correlation_filters_mobile", ""),
        })

    n = 0
    for kind, caption in FAMILY:
        grp = [x for x in renamed if x["family"] == kind]
        if not grp:
            continue
        print()
        print("%s (%d) -- %s" % (kind, len(grp), caption))
        print("-" * 104)
        for x in grp:
            n += 1
            mark = "*" if x["changed"] else " "
            print("%3d %s %-34s %s" % (n, mark, x["feature"], x["definition"][:64]))

    changed = sum(1 for x in renamed if x["changed"])
    print()
    print("%d columns, %d renamed (marked *), %d unchanged" % (n, changed, n - changed))

    if a.tex:
        emit_tex(renamed, a.tex)
        print("written: %s" % a.tex)

    if a.out:
        cols = ["family", "feature", "canonical", "changed", "definition",
                "formula", "constant",
                "survives_variance_correlation_filters_static",
                "survives_variance_correlation_filters_mobile"]
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(renamed)
        print("written: %s" % a.out)


if __name__ == "__main__":
    main()
