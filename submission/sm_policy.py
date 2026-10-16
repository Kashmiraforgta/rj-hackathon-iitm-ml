"""Supply-aware sequential matching policy, Round 2 (stdlib only, deterministic).

Uses only the observable request state: members, introductions, feedback, ask_log.
Hard constraints always go through kit.eligibility; nothing here overrides it.

Round 2 default profile 'r2':
  * every member gets the same prior (no learned propensities),
  * the daily batch is an exact maximum-weight matching (components <= 18 members),
  * edges touching members who have waited longest for a first introduction get a bonus,
  * hard-constraint asks are targeted, soft-field asks are off.
The Round 1 build is kept as profile 'r1' (learned Beta-shrunk propensities, no waiting bonus).
"""
import itertools
from kit import HARD, eligibility, baseline_asks, baseline_match

PRIORS = dict(resp=(0.8, 4.0), yes=(0.4, 4.0), sec=(0.53, 3.0))   # (prior mean, prior strength)
PROFILES = {
    'r2': dict(prior='constant', lam=2.0, wait_cap=30, goal_mult=1.5, soft_asks=False, tau=0.0,
               min_deg=0.4, alloc='exact', asks='targeted', priors=PRIORS),
    'r1': dict(prior='learned', lam=0.0, wait_cap=30, goal_mult=1.5, soft_asks=False, tau=0.0,
               min_deg=0.4, alloc='exact', asks='targeted', priors=PRIORS),
}
CFG = PROFILES['r2']
EXACT_LIMIT = 18  # vertices per component solved exactly


def make_cfg(profile='r2', **kw):
    c = dict(PROFILES[profile])
    c.update(kw)
    return c


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


def _post(succ, n, prior):
    p0, k = prior
    return (succ + k * p0) / (n + k)


def member_factors(s, pri=PRIORS):
    """Posterior means: (responds before deadline, says Yes when responding, second-meeting Yes)."""
    s = s or dict(resp=0, miss=0, yes=0, s_yes=0, s_no=0)
    return (_post(s['resp'], s['resp'] + s['miss'], pri['resp']),
            _post(s['yes'], s['resp'], pri['yes']),
            _post(s['s_yes'], s['s_yes'] + s['s_no'], pri['sec']))


def member_score(s, cfg=CFG):
    r, a, q = member_factors(s, cfg['priors'])
    return r * a * q


def predict_mutual(sa, sb, pri=PRIORS):
    """Predicted probability that an introduction ends in two Yes answers (diagnostic only)."""
    ra, ya, _ = member_factors(sa, pri)
    rb, yb, _ = member_factors(sb, pri)
    return ra * ya * rb * yb


def past_pairs(state):
    return {tuple(sorted((i['user_a'], i['user_b']))) for i in state.get('introductions', [])}


def choose_asks(state, cfg=CFG):
    if cfg['asks'] == 'none':
        return []
    if cfg['asks'] == 'baseline':
        return baseline_asks(state)
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


def max_weight_matching(edges, greedy=False):
    """edges: {(a,b): weight>0}. Exact per small component, greedy for very large ones (or if greedy=True)."""
    adj, comps = _components(edges)
    out = []
    for comp in comps:
        out += list(_exact(comp, adj) if (len(comp) <= EXACT_LIMIT and not greedy) else _heuristic(comp, adj))
    return [list(p) for p in out]


def waiting_bonus(state, cfg):
    """s_i in [0,1]: days since arrival (capped) for members never yet introduced, else 0."""
    if cfg['lam'] <= 0:
        return {}
    served = set()
    for i in state.get('introductions', []):
        served.update((i['user_a'], i['user_b']))
    day, cap = state['day'], cfg['wait_cap']
    return {m['member_id']: min(max(day - m['arrived_day'], 0), cap) / cap
            for m in state['members'] if m['available'] and m['member_id'] not in served}


def choose_pairs(state, cfg=CFG):
    ms = [m for m in state['members'] if m['available']]
    past = past_pairs(state)
    if cfg['prior'] == 'learned':
        stats = member_stats(state)
        f = {m['member_id']: member_score(stats.get(m['member_id']), cfg) for m in ms}
    else:
        f = {m['member_id']: 1.0 for m in ms}
    wb, lam = waiting_bonus(state, cfg), cfg['lam']
    edges = {}
    for a, b in itertools.combinations(ms, 2):
        key = tuple(sorted((a['member_id'], b['member_id'])))
        if key in past or eligibility(a, b)['status'] != 'feasible':
            continue
        w = f[key[0]] * f[key[1]] * (1.0 + lam * (wb.get(key[0], 0.0) + wb.get(key[1], 0.0)))
        ga, gb = a['fields']['relationship_goal'], b['fields']['relationship_goal']
        if ga and gb and ga == gb:
            w *= cfg['goal_mult']
        if w >= cfg['tau'] and w > 0:
            edges[key] = w
    return max_weight_matching(edges, greedy=(cfg['alloc'] == 'greedy'))


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


def decide(request, cfg=None):
    """Never raises: on any internal error fall back to the supplied baseline logic (still validated)."""
    cfg = cfg or CFG
    state, memory = request['state'], request.get('memory') or {}
    if request['phase'] == 'ask':
        try:
            asks = choose_asks(state, cfg)
        except Exception:
            asks = baseline_asks(state)
        return {'asks': _valid_asks(state, asks), 'memory': memory}
    try:
        pairs = choose_pairs(state, cfg)
    except Exception:
        pairs = baseline_match(state)
    return {'pairs': _valid_pairs(state, pairs), 'memory': memory}
