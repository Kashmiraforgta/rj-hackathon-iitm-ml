"""Paired analysis helpers: python r2_analyze.py file.jsonl ref policy1,policy2 [metrics]"""
import json, random, statistics, sys, collections
MET = ['msmi', 'mutual_acceptances', 'dates', 'left_waiting', 'mean_wait', 'coverage', 'ask_cost', 'assignments']
def load(*files):
    rows = {}
    for f in files:
        for l in open(f):
            if l.strip():
                d = json.loads(l); rows[(d['policy'], d['scenario'], d['seed'])] = d
    return rows
def paired(rows, a, b, m, scen=None, B=2000, seed=1):
    keys = sorted({(s, sd) for (p, s, sd) in rows if p == a and (p == a) and (b, s, sd) in rows and (scen is None or s in scen)})
    d = [rows[(a, s, sd)][m] - rows[(b, s, sd)][m] for s, sd in keys]
    if not d: return None
    r = random.Random(seed); n = len(d); bs = sorted(sum(d[r.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(d) / n, bs[int(.025 * B)], bs[int(.975 * B)], n
def fmt(t, nd=2): return 'n/a' if t is None else f'{t[0]:+.{nd}f} [{t[1]:+.{nd}f},{t[2]:+.{nd}f}]'
def mean(rows, p, m, scen=None):
    v = [d[m] for (pp, s, sd), d in rows.items() if pp == p and (scen is None or s in scen)]
    return sum(v) / len(v) if v else None
if __name__ == '__main__':
    rows = load(sys.argv[1]); ref = sys.argv[2]; pols = sys.argv[3].split(',')
    mets = sys.argv[4].split(',') if len(sys.argv) > 4 else MET
    print('policy'.ljust(12), 'n', *[m[:14].ljust(22) for m in mets])
    for p in pols:
        t = [paired(rows, p, ref, m) for m in mets]
        print(p.ljust(12), t[0][3] if t[0] else 0, *[fmt(x).ljust(22) for x in t])
