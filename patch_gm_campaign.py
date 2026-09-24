#!/usr/bin/env python3
"""
Create run_campaign_gaussmarkov.sh from the canonical run_campaign_v347.sh.

Three edits, each anchored on an exact string that must be present exactly once.
If any anchor is missing or ambiguous the script aborts and writes nothing --
the canonical campaign script is never modified, only read.

  1. OUT  -> simulations_v347_gaussmarkov   (own tree, cannot collide)
  2. the write guard is retargeted to that tree
  3. --mobilityModel=gaussmarkov AND --enforceHopFilter=0 are added

⛔ On --enforceHopFilter=0. The distance gate is OFF, and it must stay off.
run_campaign_v347.sh does not pass the flag, so on its own it defaults to ON --
but the campaign this arm has to be comparable with, the 10,000-run C_all
listener population, is the UNRESTRICTED one: the base campaign plus the runs
recovered by run_nohopfilter.sh / run_nohopfilter_mobile.sh, whose header says
"producing the missing part of the unrestricted population". STATE 25.47-B
confirms it ("the distance-recovered runs are all included"), and 17.15 records
why the condition was dropped: it selects for runs in which the ATTACK damages
one particular flow, which is not what this paper measures. Leaving it on here
would build a different population from the control and silently break the
comparison.

Idempotent: re-running overwrites the generated file from the canonical source.
"""
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "run_campaign_v347.sh")
DST = os.path.join(ROOT, "run_campaign_gaussmarkov.sh")

EDITS = [
    (
        'OUT="$ROOT/simulations_v347"',
        'OUT="$ROOT/simulations_v347_gaussmarkov"',
    ),
    (
        'case "$OUT" in */simulations_v347) : ;; *) echo "REFUSING: bad OUT $OUT"; exit 1 ;; esac',
        'case "$OUT" in */simulations_v347_gaussmarkov) : ;; *) echo "REFUSING: bad OUT $OUT"; exit 1 ;; esac',
    ),
    (
        '            --run="$tmp" --RngRun="$seed" --bMobility="$MOB" \\',
        '            --run="$tmp" --RngRun="$seed" --bMobility="$MOB" \\\n'
        '            --mobilityModel=gaussmarkov --enforceHopFilter=0 \\',
    ),
    (
        # The banner must describe what this variant actually admits. Left as
        # inherited it says the distance condition is applied, and anyone
        # reading the log later would conclude the wrong population was built.
        'echo "accept = fully connected AND sender >=3 hops from victim"',
        'echo "accept = fully connected at t=60 (no distance condition -- C_all population)"\n'
        'echo "mobility = Gauss-Markov (alpha 0.85, TimeStep 3 s, mean speed 1.5-2.0 m/s)"',
    ),
]

with open(SRC, encoding="utf-8") as fh:
    text = fh.read()

for old, new in EDITS:
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT: anchor found {n} times, expected exactly 1:\n{old!r}")
    text = text.replace(old, new)

# Belt and braces: no EXECUTABLE line may still name the canonical tree.
# Comments are exempt -- the header prose legitimately describes the original.
bad = []
for ln in text.splitlines():
    code = ln.split("#", 1)[0]
    if "simulations_v347\"" in code or "simulations_v347/" in code \
       or "simulations_v347)" in code:
        bad.append(ln)
if bad:
    sys.exit("ABORT: generated script still references the canonical tree:\n"
             + "\n".join(bad))

if "--mobilityModel=gaussmarkov" not in text:
    sys.exit("ABORT: mobility flag not present after patching")

with open(DST, "w", encoding="utf-8") as fh:
    fh.write(text)
os.chmod(DST, 0o755)

print(f"wrote {DST}")
print("  OUT tree      : simulations_v347_gaussmarkov")
print("  mobility model: gaussmarkov")
print(f"  canonical script untouched: {SRC}")
