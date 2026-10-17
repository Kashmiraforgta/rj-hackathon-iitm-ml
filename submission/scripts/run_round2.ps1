python r2_eval.py --out results\stageA.jsonl --seeds 3001-3009 --policies greedy,r1,const,r2,r2_lam1,r2_lam4,r2_greedy,r2_basask,r2_noask,r2_soft,r2_learned,oracle
python r2_fitpriors.py
python r2_eval.py --out results\stageC.jsonl --seeds 6001-6006 --policies greedy,no_asks,random,r1,const,r2,r2_learned,r2_learncal,r2_soft,r2_greedy,r2_basask,oracle --priors results\priors_cal.json
python r2_eval.py --out results\stageC_proto.jsonl --seeds 6001-6006 --policies proto
python r2_eval.py --out results\stress.jsonl --seeds 7001-7004 --scenarios miss_high,resp_low,resp_drop,resp_het,accept_het --policies greedy,r1,const,r2,r2_learned,r2_learncal,r2_soft,r2_basask,r2_noask,oracle --priors results\priors_cal.json
python r2_analyze.py results\stageC.jsonl greedy r1,const,r2,r2_learned
python evaluate.py --seeds 101 --variants all --output results\official_r2_seed101.json
