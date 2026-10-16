"""Prototype v0: value-of-information asks + propensity-weighted global matching. Observable state only."""
import itertools, math
import networkx as nx
from kit import HARD, eligibility

def member_stats(state):
    st = {}
    def g(i): return st.setdefault(i, dict(resp=0, miss=0, yes=0, no=0, s_yes=0, s_no=0))
    for e in state['feedback']:
        i = e.get('member_id')
        if i is None: continue
        if e['event'] == 'introduction_response':
            if e['value'] is None: g(i)['miss'] += 1
            else: g(i)['resp'] += 1; g(i)['yes' if e['value']=='yes' else 'no'] += 1
        elif e['event'] == 'second_meeting_intention' and e['value'] is not None:
            g(i)['s_yes' if e['value']=='yes' else 's_no'] += 1
    return st

def score_member(s, cfg):
    r = (s['resp'] + 3.2) / (s['resp'] + s['miss'] + 4)
    a = (s['yes'] + 1.6) / (s['resp'] + 4)
    q = (s['s_yes'] + 1.6) / (s['s_yes'] + s['s_no'] + 3)
    return (r ** cfg['wr']) * (a ** cfg['wa']) * (q ** cfg['wq'])

def asks(state, cfg):
    day = state['day']; budget = state['ask_budget_remaining']
    ms = [m for m in state['members'] if m['available']]
    past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
    deg = {}
    cand = [m for m in ms if not any(m['field_status'][k]=='declined' for k in HARD)]
    cset = {m['member_id'] for m in cand}
    for a, b in itertools.combinations(cand, 2):
        e = eligibility(a, b)
        if e['status'] == 'infeasible' or tuple(sorted((a['member_id'], b['member_id']))) in past: continue
        for x, y in ((a, b), (b, a)):
            if any(x['fields'][k] is None for k in HARD):
                # value higher when the partner is already fully known (one ask unlocks the pair)
                unlock = 1.0 if not any(y['fields'][k] is None for k in HARD) else 0.4
                deg[x['member_id']] = deg.get(x['member_id'], 0) + unlock
    out = []
    for i, v in sorted(deg.items(), key=lambda t: -t[1]):
        if budget >= 3 and v >= cfg['min_deg']:
            out.append({'member_id': i, 'field': 'constraints'}); budget -= 3
    if cfg['soft']:
        for m in sorted(ms, key=lambda m: m['member_id']):
            if budget <= 0: break
            if m['field_status']['relationship_goal'] == 'not_asked' and m['member_id'] in deg or \
               (m['field_status']['relationship_goal'] == 'not_asked' and not any(m['fields'][k] is None for k in HARD)):
                out.append({'member_id': m['member_id'], 'field': 'relationship_goal'}); budget -= 1
    return out

def match(state, cfg):
    ms = [m for m in state['members'] if m['available']]
    past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
    stats = member_stats(state)
    f = {m['member_id']: score_member(stats.get(m['member_id'], dict(resp=0,miss=0,yes=0,no=0,s_yes=0,s_no=0)), cfg) for m in ms}
    G = nx.Graph()
    for a, b in itertools.combinations(ms, 2):
        key = tuple(sorted((a['member_id'], b['member_id'])))
        if key in past or eligibility(a, b)['status'] != 'feasible': continue
        w = f[a['member_id']] * f[b['member_id']]
        ga, gb = a['fields']['relationship_goal'], b['fields']['relationship_goal']
        if ga and gb: w *= cfg['goal_mult'] if ga == gb else 1.0
        if w < cfg['tau']: continue
        G.add_edge(*key, weight=w if cfg['global'] else w)
    if cfg['global']:
        pairs = [list(p) for p in nx.max_weight_matching(G, maxcardinality=cfg['maxcard'])]
    else:
        used, pairs = set(), []
        for a, b, d in sorted(G.edges(data=True), key=lambda t: -t[2]['weight']):
            if a not in used and b not in used: pairs.append([a, b]); used.update((a, b))
    return pairs

DEFAULT = dict(wr=1.0, wa=1.0, wq=1.0, min_deg=0.4, soft=True, goal_mult=1.5, tau=0.0, **{'global': True}, maxcard=False)
