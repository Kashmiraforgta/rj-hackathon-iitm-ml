#!/bin/sh
# Reproduces all four required result files on identical seeds and scenarios (about 45 min on 1 core).
for b in sm r1 greedy no_asks random; do
  python3 evaluate.py --baseline $b --seeds 101,102,103 --variants all --output results/final_$b.json > results/final_$b.log 2>&1
done
