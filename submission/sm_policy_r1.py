"""Supply-aware sequential matching policy (stdlib only, deterministic).

Uses only the observable request state: members, introductions, feedback, ask_log.
Hard constraints always go through kit.eligibility; nothing here overrides it.
"""
import itertools
from kit import HARD, eligibility, baseline_asks, baseline_match

CFG = dict(wr=1.0, wa=1.0, wq=1.0, min_deg=0.4, soft_asks=False, goal_mult=1.5, tau=0.0)
EXACT_LIMIT = 18  # vertices per component solved exactly


def member_stats(state):
    st = {}
    for e in state.get('feedback', []):
        i = e.get('member_id')
        if i is None:
            continue
        s = st.setdefault(i, dict(resp=0, miss=0, yes=0, s_yes=0, s_no=0))
        if e['event'] == 'introduction_response':
            if e['value'] is None:
                s['miss'] += 1
            else:
                s['resp'] += 1
                s['yes'] += e['value'] == 'yes'
        elif e['event'] == 'second_meeting_intention' and e['value'] is not None:
            s['s_yes' if e['value'] == 'yes' else 's_no'] += 1
    return st


def member_score(s, cfg=CFG):
    s = s or dict(resp=0, miss=0, yes=0, s_yes=0, s_no=0)
    r = (s['resp'] + 3.2) / (s['resp'] + s['miss'] + 4)       # responds before deadline (Beta prior .8)
    a = (s['yes'] + 1.6) / (s['resp'] + 4)                    # says Yes when responding (prior .4)
    q = (s['s_yes'] + 1.6) / (s['s_yes'] + s['s_no'] + 3)     # second-meeting Yes (prior .53)
    return r ** cfg['wr'] * a ** cfg['wa'] * q ** cfg['wq']


def past_pairs(state):
    return {tuple(sorted((i['user_a'], i['user_b']))) for i in state.get('introductions', [])}


def choose_asks(state, cfg=CFG):
    budget = state['ask_budget_remaining']
    ms = [m for m in state['members'] if m['available']
          and not any(m['field_status'][k] == 'declined' for k in HARD)]
    past, deg = past_pairs(state), {}
    for a, b in itertools.combinations(ms, 2):
        if tuple(sorted((a['member_id'], b['member_id']))) in past:
            continue
        if eligibility(a, b)['status'] == 'infeasible':
            continue
        for x, y in ((a, b), (b, a)):
            if any(x['fields'][k] is None for k in HARD):
                unlock = 0.4 if any(y['fields'][k] is None for k in HARD) else 1.0
                deg[x['member_id']] = deg.get(x['member_id'], 0.0) + unlock
    out = []
    for i, v in sorted(deg.items(), key=lambda t: (-t[1], t[0])):
        if budget >= 3 and v >= cfg['min_deg']:
            out.append({'member_id': i, 'field': 'constraints'})
            budget -= 3
    if cfg['soft_asks']:
        for m in sorted(ms, key=lambda m: m['member_id']):
            if budget <= 0:
                break
            if m['field_status']['relationship_goal'] == 'not_asked' and not any(m['fields'][k] is None for k in HARD):
                out.append({'member_id': m['member_id'], 'field': 'relationship_goal'})
                budget -= 1
    return out


def _components(edges):
    adj = {}
    for (a, b), w in edges.items():
        adj.setdefault(a, {})[b] = w
        adj.setdefault(b, {})[a] = w
    seen, comps = set(), []
    for v in sorted(adj):
        if v in seen:
            continue
        stack, comp = [v], []
        seen.add(v)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(sorted(comp))
    return adj, comps


def _exact(vs, adj):
    memo = {}

    def solve(rem):
        if len(rem) < 2:
            return 0.0, ()
        if rem in memo:
            return memo[rem]
        v = min(rem)
        best = solve(rem - {v})
        for u, w in adj[v].items():
            if u in rem:
                sub = solve(rem - {v, u})
                if sub[0] + w > best[0]:
                    best = (sub[0] + w, sub[1] + ((v, u),))
        memo[rem] = best
        return best
    return solve(frozenset(vs))[1]


def _heuristic(vs, adj):
    edges = sorted(((w, a, b) for a in vs for b, w in adj[a].items() if a < b), key=lambda t: (-t[0], t[1], t[2]))
    used, pairs = set(), []
    for w, a, b in edges:
        if a not in used and b not in used:
            pairs.append((a, b))
            used.update((a, b))
    return tuple(pairs)


def max_weight_matching(edges):
    """edges: {(a,b): weight>0}. Exact per small component, greedy for very large ones."""
    adj, comps = _components(edges)
    out = []
    for comp in comps:
        out += list(_exact(comp, adj) if len(comp) <= EXACT_LIMIT else _heuristic(comp, adj))
    return [list(p) for p in out]


def choose_pairs(state, cfg=CFG):
    ms = [m for m in state['members'] if m['available']]
    past, stats = past_pairs(state), member_stats(state)
    f = {m['member_id']: member_score(stats.get(m['member_id']), cfg) for m in ms}
    edges = {}
    for a, b in itertools.combinations(ms, 2):
        key = tuple(sorted((a['member_id'], b['member_id'])))
        if key in past or eligibility(a, b)['status'] != 'feasible':
            continue
        w = f[key[0]] * f[key[1]]
        ga, gb = a['fields']['relationship_goal'], b['fields']['relationship_goal']
        if ga and gb and ga == gb:
            w *= cfg['goal_mult']
        if w >= cfg['tau'] and w > 0:
            edges[key] = w
    return max_weight_matching(edges)


def _valid_pairs(state, pairs):
    byid = {m['member_id']: m for m in state['members']}
    past, used, out = past_pairs(state), set(), []
    for a, b in pairs:
        key = tuple(sorted((a, b)))
        if a == b or a in used or b in used or key in past or a not in byid or b not in byid:
            continue
        if not (byid[a]['available'] and byid[b]['available']) or eligibility(byid[a], byid[b])['status'] != 'feasible':
            continue
        used.update((a, b))
        out.append([a, b])
    return out


def _valid_asks(state, asks):
    byid = {m['member_id']: m for m in state['members']}
    budget, seen, out = state['ask_budget_remaining'], set(), []
    for a in asks:
        cost = 3 if a['field'] == 'constraints' else 1
        if a['member_id'] in byid and byid[a['member_id']]['available'] and (a['member_id'], a['field']) not in seen and cost <= budget:
            budget -= cost
            seen.add((a['member_id'], a['field']))
            out.append(a)
    return out


def decide(request):
    """Never raises: on any internal error fall back to the supplied baseline logic (still validated)."""
    state, memory = request['state'], request.get('memory') or {}
    if request['phase'] == 'ask':
        try:
            asks = choose_asks(state)
        except Exception:
            asks = baseline_asks(state)
        return {'asks': _valid_asks(state, asks), 'memory': memory}
    try:
        pairs = choose_pairs(state)
    except Exception:
        pairs = baseline_match(state)
    return {'pairs': _valid_pairs(state, pairs), 'memory': memory}
