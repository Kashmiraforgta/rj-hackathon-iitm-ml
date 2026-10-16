"""Round 2 in-process paired evaluation. Resumable: appends one JSON line per (policy, scenario, seed).
Usage: python r2_eval.py --stage A|C|S --out results/stageA.jsonl [--policies a,b] [--seeds 3001-3009]
Policies see only observable state (the 'oracle' reference additionally reads simulator truth, upper bound only)."""
import argparse, itertools, json, math, random, sys, time
from pathlib import Path
import kit, sm_policy as sp
from kit import Simulator, generate, HARD, baseline_asks, baseline_match, eligibility as _elig

_cache = {}
def cached_elig(a, b):
    """Exact memo of kit.eligibility (key changes whenever either member's observed fields change)."""
    k = (a['member_id'], b['member_id'], sum(v is not None for v in a['fields'].values()),
         sum(v is not None for v in b['fields'].values()))
    r = _cache.get(k)
    if r is None:
        r = _cache[k] = _elig(a, b)
    return r
sp.eligibility = cached_elig
try:
    import proto; proto.eligibility = cached_elig
except ImportError:
    proto = None

VARIANTS = ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')
STRESS = ('miss_high', 'resp_low', 'resp_drop', 'resp_het', 'accept_het')

class DropSim(Simulator):          # response behaviour changes at day 30 (learned stats go stale)
    def advance(self, pairs):
        if self.day == 30:
            for m in self.members.values(): m['response_rate'] *= 0.5
        super().advance(pairs)

def make_world(seed, scen):
    if scen in VARIANTS:
        return generate(seed, 200, 'evaluation', scen), Simulator
    w = generate(seed, 200, 'evaluation', 'development'); r = random.Random(seed * 7919 + 13)
    for m in w['members']:
        if scen == 'miss_high':   # 70% of observed answers disappear, 15% of the rest are declined
            for k in list(m['fields']):
                if m['fields'][k] is not None and r.random() < .7:
                    m['fields'][k] = None; m['field_status'][k] = 'not_asked'; m['field_observed_day'][k] = None
                if m['fields'][k] is None and m['field_status'][k] == 'not_asked' and r.random() < .15:
                    m['field_status'][k] = 'declined'
        elif scen == 'resp_low': m['response_rate'] *= .6
        elif scen == 'resp_het': m['response_rate'] = r.choice([.2, .95])
        elif scen == 'accept_het': m['bias'] = r.gauss(0, 1.6); m['second_bias'] = r.gauss(0, 1.2)
    return w, (DropSim if scen == 'resp_drop' else Simulator)

# ---- policies: name -> (cfg or special) ---------------------------------------------------------
def C(profile='r2', **kw): return sp.make_cfg(profile, **kw)
CAL = None   # calibrated priors, filled by --priors file
POLICIES = {
 'greedy':      'greedy', 'no_asks': 'no_asks', 'random': 'random',
 'r1':          lambda: C('r1'),
 'const':       lambda: C('r2', lam=0.0),                       # constant prior, matching, no waiting bonus
 'r2':          lambda: C('r2'),                                # Round 2 default
 'r2_lam1':     lambda: C('r2', lam=1.0),
 'r2_lam4':     lambda: C('r2', lam=4.0),
 'r2_lam8':     lambda: C('r2', lam=8.0),
 'r2_greedy':   lambda: C('r2', alloc='greedy'),
 'r2_basask':   lambda: C('r2', asks='baseline'),
 'r2_noask':    lambda: C('r2', asks='none'),
 'r2_soft':     lambda: C('r2', soft_asks=True),
 'r2_nogoal':   lambda: C('r2', goal_mult=1.0),
 'r2_learned':  lambda: C('r2', prior='learned'),
 'r2_learncal': lambda: C('r2', prior='learned', priors=CAL),
 'learncal':    lambda: C('r1', priors=CAL),
 'oracle':      'oracle',
 'proto':       'proto',
}

def true_p_mutual(w, a, b, day):
    A, B = w[a], w[b]; z = lambda x, y: 1 / (1 + math.exp(-(-.25 + x['bias'] + sum(wt * (1 if x['truth'][k] == y['truth'][k] else -.5) for k, wt in
              {'relationship_goal': .7, 'relationship_pace': .4, 'lifestyle': .25, 'conversations': .2}.items()))))
    return A['response_rate'] * B['response_rate'] * z(A, B) * z(B, A)

_truth_cache = {}
def truth_graph(world, key):
    if key in _truth_cache: return _truth_cache[key]
    ms = [dict(m, fields=dict(m['truth'])) for m in world['members']
          if not any(m['field_status'][k] == 'declined' for k in HARD) and m['arrived_day'] <= 59]
    deg = {m['member_id']: 0 for m in ms}
    for a, b in itertools.combinations(ms, 2):
        if _elig(a, b)['status'] == 'feasible': deg[a['member_id']] += 1; deg[b['member_id']] += 1
    _truth_cache[key] = deg; return deg

def run(world, simcls, pol, key):
    sim = simcls(world); spec = POLICIES[pol]; cfg = spec() if callable(spec) else None
    byw = {m['member_id']: m for m in sim.world['members']}
    preds = []   # (uncalibrated prediction, calibrated prediction, outcome-later) per introduction
    for _ in range(60):
        s = sim.observe()
        if spec == 'greedy': asks = baseline_asks(s)
        elif spec == 'no_asks': asks = []
        elif spec == 'random': asks = baseline_asks(s)
        elif spec == 'oracle': asks = baseline_asks(s)
        elif spec == 'proto': asks = proto.asks(s, proto.DEFAULT)
        else: asks = sp.decide({'phase': 'ask', 'state': s, 'memory': None}, cfg)['asks']
        sim.resolve_asks(asks); s = sim.observe()
        if spec in ('greedy', 'no_asks'): pairs = baseline_match(s)
        elif spec == 'random':
            c = [m for m in s['members'] if m['available']]; past = sp.past_pairs(s)
            ed = [tuple(sorted((a['member_id'], b['member_id']))) for a, b in itertools.combinations(c, 2)
                  if cached_elig(a, b)['status'] == 'feasible' and tuple(sorted((a['member_id'], b['member_id']))) not in past]
            random.Random(17 + s['day']).shuffle(ed); pairs, used = [], set()
            for p in ed:
                if not used.intersection(p): pairs.append(list(p)); used.update(p)
        elif spec == 'proto': pairs = sp._valid_pairs(s, proto.match(s, proto.DEFAULT))
        elif spec == 'oracle':
            ms = [m for m in s['members'] if m['available']]; past = sp.past_pairs(s); edges = {}
            for a, b in itertools.combinations(ms, 2):
                k = tuple(sorted((a['member_id'], b['member_id'])))
                if k not in past and cached_elig(a, b)['status'] == 'feasible': edges[k] = true_p_mutual(byw, k[0], k[1], s['day']) + 1e-6
            pairs = sp._valid_pairs(s, sp.max_weight_matching(edges))
        else: pairs = sp.decide({'phase': 'match', 'state': s, 'memory': None}, cfg)['pairs']
        if pairs:   # diagnostic predictions with the learned estimator (policy-agnostic), at assignment time
            st = sp.member_stats(s)
            for a, b in pairs:
                preds.append((sp.predict_mutual(st.get(a), st.get(b)),
                              sp.predict_mutual(st.get(a), st.get(b), CAL) if CAL else None,
                              tuple(sorted((a, b)))))
        sim.advance(pairs)
    arrived = {m['member_id']: m['arrived_day'] for m in sim.observe()['members'] if m['arrived_day'] <= 59}
    first = {}
    for i in sim.introductions:
        for m in (i['user_a'], i['user_b']): first.setdefault(m, i['assigned_day'])
    for _ in range(40): sim.advance([])
    r = sim.metrics()
    ev = {}
    for e in sim.events:
        if e['event'] == 'introduction_response': ev.setdefault(e['introduction_id'], []).append(e['value'])
    iid = {tuple(sorted((i['user_a'], i['user_b']))): i['introduction_id'] for i in sim.introductions}
    cal = []
    for pu, pc, k in preds:
        v = ev.get(iid[k], []); cal.append([round(pu, 5), None if pc is None else round(pc, 5), int(len(v) == 2 and all(x == 'yes' for x in v))])
    deg = truth_graph(world, key)
    servable = [m for m in deg if deg[m] > 0]
    waits = [first[m] - arrived[m] for m in first if m in arrived]
    n = len(arrived)
    r.update(n=n, coverage=len(set(first) & set(arrived)) / n, msmi=r['mutual_second_meeting_intention'],
             mutual100=100 * r['mutual_acceptances'] / n, servable=len(servable),
             left_waiting=sum(m not in first for m in servable),
             mean_wait=sum(waits) / max(1, len(waits)), p90_wait=sorted(waits)[int(.9 * (len(waits) - 1))] if waits else 0,
             unserved=n - len(set(first) & set(arrived)), cal=cal)
    return r

def parse_seeds(s):
    out = []
    for part in s.split(','):
        if '-' in part: a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--policies', required=True)
    ap.add_argument('--seeds', required=True); ap.add_argument('--scenarios', default=','.join(VARIANTS))
    ap.add_argument('--priors'); ap.add_argument('--limit_sec', type=float, default=1e9)
    a = ap.parse_args()
    if a.priors: CAL = json.load(open(a.priors))
    out = Path(a.out); done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip(): d = json.loads(l); done.add((d['policy'], d['scenario'], d['seed']))
    t0 = time.time()
    for sd in parse_seeds(a.seeds):
        for sc in a.scenarios.split(','):
            todo = [p for p in a.policies.split(',') if (p, sc, sd) not in done]
            if not todo: continue
            w, cls = make_world(sd, sc)
            for p in todo:
                if time.time() - t0 > a.limit_sec: sys.exit(0)
                r = run(w, cls, p, (sd, sc)); r.update(policy=p, scenario=sc, seed=sd)
                with out.open('a') as f: f.write(json.dumps(r) + '\n')
                print(p, sc, sd, round(r['msmi'], 1), round(r['mutual100'], 2), round(time.time() - t0), flush=True)
