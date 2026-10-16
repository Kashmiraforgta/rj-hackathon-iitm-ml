import sys, json, statistics, collections
from kit import Simulator, generate
import proto
from kit import baseline_asks, baseline_match
VARS=('development','sparse','cold_start','delayed','shift','drift')
def run(world, ask_fn, match_fn):
    sim=Simulator(world)
    for _ in range(60):
        sim.resolve_asks(ask_fn(sim.observe())); sim.advance(match_fn(sim.observe()))
    arrived={m['member_id'] for m in sim.observe()['members'] if m['arrived_day']<=59}
    first=set()
    for i in sim.introductions: first.update((i['user_a'],i['user_b']))
    for _ in range(40): sim.advance([])
    r=sim.metrics(); r['coverage']=len(first&arrived)/len(arrived); r['n']=len(arrived)
    r['mutual_per100']=100*r['mutual_acceptances']/len(arrived); return r
def policies(cfg):
    return {'greedy_baseline':(baseline_asks,baseline_match),
            'v0':(lambda s:proto.asks(s,cfg),lambda s:proto.match(s,cfg))}
def main(seeds, variants, pols):
    res=collections.defaultdict(lambda: collections.defaultdict(list))
    for v in variants:
        for sd in seeds:
            w=generate(sd,200,'evaluation',v)
            for name,(a,m) in pols.items(): res[name][v].append(run(w,a,m))
    out={}
    for name,d in res.items():
        fam={v:{k:statistics.mean(r[k] for r in rs) for k in ('msmi_per_100_arrived_members','coverage','mutual_per100','dates','assignments','ask_cost')} for v,rs in d.items()}
        out[name]={'score':statistics.mean(f['msmi_per_100_arrived_members'] for f in fam.values()),
                   'coverage':statistics.mean(f['coverage'] for f in fam.values()),
                   'mutual100':statistics.mean(f['mutual_per100'] for f in fam.values()),
                   'dates':statistics.mean(f['dates'] for f in fam.values()),
                   'assign':statistics.mean(f['assignments'] for f in fam.values()),
                   'ask':statistics.mean(f['ask_cost'] for f in fam.values()),'fam':fam}
    return out
if __name__=='__main__':
    seeds=list(range(int(sys.argv[1]),int(sys.argv[2])))
    o=main(seeds,VARS,policies(proto.DEFAULT))
    for n,v in o.items(): print(n,{k:round(x,3) for k,x in v.items() if k!='fam'})
