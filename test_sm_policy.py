import itertools, random, unittest
from kit import HARD, Simulator, generate
import sm_policy as sp


def brute(edges):
    best = 0.0
    items = list(edges.items())
    def rec(i, used, tot):
        nonlocal best
        best = max(best, tot)
        for j in range(i, len(items)):
            (a, b), w = items[j]
            if a not in used and b not in used:
                rec(j + 1, used | {a, b}, tot + w)
    rec(0, frozenset(), 0.0)
    return best


class PolicyTests(unittest.TestCase):
    def test_empty_population(self):
        state = Simulator(generate(1, 0)).observe()
        self.assertEqual(sp.decide({'phase': 'ask', 'state': state, 'memory': None}), {'asks': [], 'memory': {}})
        self.assertEqual(sp.decide({'phase': 'match', 'state': state, 'memory': None}), {'pairs': [], 'memory': {}})

    def test_matching_is_optimal_on_small_graphs(self):
        r = random.Random(3)
        for _ in range(60):
            nodes = [f'n{i}' for i in range(r.randint(2, 8))]
            edges = {tuple(sorted(p)): r.uniform(.05, 1) for p in itertools.combinations(nodes, 2) if r.random() < .5}
            pairs = sp.max_weight_matching(edges)
            used = [x for p in pairs for x in p]
            self.assertEqual(len(used), len(set(used)))
            self.assertAlmostEqual(sum(edges[tuple(sorted(p))] for p in pairs), brute(edges))

    def test_full_episodes_are_always_valid(self):
        for variant in ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift'):
            sim = Simulator(generate(7, 120, 'eval', variant))
            for _ in range(60):
                state = sim.observe()
                sim.resolve_asks(sp.decide({'phase': 'ask', 'state': state, 'memory': None})['asks'])
                state = sim.observe()
                sim.advance(sp.decide({'phase': 'match', 'state': state, 'memory': None})['pairs'])
            self.assertGreater(len(sim.introductions), 0)

    def test_never_asks_declined_or_unavailable(self):
        sim = Simulator(generate(11, 150))
        state = sim.observe()
        byid = {m['member_id']: m for m in state['members']}
        asks = sp.decide({'phase': 'ask', 'state': state, 'memory': None})['asks']
        self.assertLessEqual(sum(3 if a['field'] == 'constraints' else 1 for a in asks), 12)
        for a in asks:
            self.assertTrue(byid[a['member_id']]['available'])
            self.assertFalse(any(byid[a['member_id']]['field_status'][k] == 'declined' for k in HARD))

    def test_internal_error_falls_back_validly(self):
        sim = Simulator(generate(5, 100))
        state = sim.observe()
        orig = sp.choose_pairs
        sp.choose_pairs = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom'))
        try:
            out = sp.decide({'phase': 'match', 'state': state, 'memory': None})
        finally:
            sp.choose_pairs = orig
        sim.advance(out['pairs'])

    def test_deterministic(self):
        state = Simulator(generate(9, 100)).observe()
        req = {'phase': 'match', 'state': state, 'memory': None}
        self.assertEqual(sp.decide(req), sp.decide(req))


if __name__ == '__main__':
    unittest.main()
