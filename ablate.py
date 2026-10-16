import sys, json, proto, fast_eval
from kit import baseline_asks
seeds=list(range(int(sys.argv[1]),int(sys.argv[2])))
base=dict(proto.DEFAULT)
def mk(**kw): 
    c=dict(base); c.update(kw); return c
cfgs={'v0_full':mk(),
      'abl_greedy_alloc':mk(**{'global':False}),
      'abl_no_propensity':mk(wr=0,wa=0,wq=0),
      'abl_no_soft_asks':mk(soft=False,goal_mult=1.0),
      'abl_baseline_asks':None}
pols={}
for n,c in cfgs.items():
    if c is None: pols[n]=(baseline_asks,lambda s,c=cfgs['v0_full']:proto.match(s,c))
    else: pols[n]=(lambda s,c=c:proto.asks(s,c),lambda s,c=c:proto.match(s,c))
pols['greedy_baseline']=fast_eval.policies(base)['greedy_baseline']
o=fast_eval.main(seeds,fast_eval.VARS,pols)
json.dump(o,open('results/ablation.json','w'),indent=1)
for n,v in o.items(): print(n,{k:round(x,3) for k,x in v.items() if k!='fam'})
