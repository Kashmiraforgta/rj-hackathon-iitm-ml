#!/bin/sh
# Reproduces Round 2 results (about 1 h on 1 core). Each command is resumable.
mkdir -p results
python3 r2_eval.py --out results/stageA.jsonl --seeds 3001-3009 --policies greedy,r1,const,r2,r2_lam1,r2_lam4,r2_greedy,r2_basask,r2_noask,r2_soft,r2_learned,oracle
python3 r2_fitpriors.py                      # writes results/priors_cal.json from seeds 3001-3009
python3 r2_eval.py --out results/stageC.jsonl --seeds 6001-6006 --policies greedy,no_asks,random,r1,const,r2,r2_learned,r2_learncal,r2_soft,r2_greedy,r2_basask,oracle --priors results/priors_cal.json
python3 r2_eval.py --out results/stageC_proto.jsonl --seeds 6001-6006 --policies proto   # needs networkx
python3 r2_eval.py --out results/stress.jsonl --seeds 7001-7004 --scenarios miss_high,resp_low,resp_drop,resp_het,accept_het --policies greedy,r1,const,r2,r2_learned,r2_learncal,r2_soft,r2_basask,r2_noask,oracle --priors results/priors_cal.json
python3 r2_analyze.py results/stageC.jsonl greedy r1,const,r2,r2_learned
python3 evaluate.py --seeds 101 --variants all --output results/official_r2_seed101.json
