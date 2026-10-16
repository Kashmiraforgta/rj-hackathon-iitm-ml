"""Empirical-Bayes calibration of the Beta priors on stage-A (seeds 3001-3009) feedback only."""
import json, math, time, sys
import numpy as np
from scipy.special import betaln, gammaln
from scipy.optimize import minimize
import r2_eval as E, sm_policy as sp
from kit import baseline_asks, baseline_match

def collect(seed, scen):
    w, cls = E.make_world(seed, scen); sim = cls(w)
    for _ in range(60):
        sim.resolve_asks(baseline_asks(sim.observe())); sim.advance(baseline_match(sim.observe()))
    for _ in range(40): sim.advance([])
    st = sp.member_stats({'feedback': sim.events}); return st

def bb_fit(ns, ss):
    ns, ss = np.array(ns, float), np.array(ss, float)
    def nll(t):
        p = 1 / (1 + math.exp(-t[0])); k = math.exp(t[1]); a, b = p * k, (1 - p) * k
        return -np.sum(betaln(ss + a, ns - ss + b) - betaln(a, b))
    best = min((minimize(nll, [x, y], method='Nelder-Mead') for x in (-1, 0, 1) for y in (0, 1.5, 3)), key=lambda r: r.fun)
    return 1 / (1 + math.exp(-best.x[0])), math.exp(best.x[1])

if __name__ == '__main__':
    t0 = time.time(); R, Y, S = [], [], []
    for sd in range(3001, 3010):
        for sc in E.VARIANTS:
            for m in collect(sd, sc).values():
                if m['resp'] + m['miss'] > 0: R.append((m['resp'] + m['miss'], m['resp']))
                if m['resp'] > 0: Y.append((m['resp'], m['yes']))
                if m['s_yes'] + m['s_no'] > 0: S.append((m['s_yes'] + m['s_no'], m['s_yes']))
        print(sd, round(time.time() - t0), flush=True)
    out = {}
    for name, d in (('resp', R), ('yes', Y), ('sec', S)):
        p, k = bb_fit([x[0] for x in d], [x[1] for x in d])
        out[name] = [round(p, 4), round(max(1.0, min(k, 50.0)), 2)]
        print(name, 'members', len(d), 'pooled', round(sum(x[1] for x in d) / sum(x[0] for x in d), 4), 'fit', out[name])
    json.dump(out, open('results/priors_cal.json', 'w'), indent=1)
