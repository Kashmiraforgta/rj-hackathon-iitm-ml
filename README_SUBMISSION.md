# Supply-aware sequential matching policy (Round 2 build)

Solo entry: Kashmira Matale. Built on the public kit `1.0.0` (commit `a8e26b35118c`).

## What it does

`sm_policy.py` (standard library only, deterministic) implements the two-phase JSON protocol.

- **Feasibility:** every pair goes through the supplied `kit.eligibility`; only `feasible` pairs enter the allocation graph. Members with a declined hard field are never asked or matched.
- **Clarification:** each day, ask the hard-constraint bundle for the members with the most optimistically feasible partners (partners already fully known count more). Soft-field asks are **off** by default (`CFG['soft_asks']`) because the Round 1 ablation showed extra cost and no distinguishable benefit.
- **Allocation:** edge weight = product of both members' shrunken estimates of (responds before deadline) x (says Yes) x (second-meeting Yes), computed only from feedback already observed, times 1.5 if both observed `relationship_goal` match. A maximum-weight matching picks the daily batch: exact for components up to 18 members, greedy beyond that.
- **Waiting:** any feasible pair is introduced (`tau = 0`); an empty graph returns an empty list.
- **Safety:** output pairs and asks are re-validated (available, distinct, no repeats, within budget), and any internal error falls back to the supplied baseline logic, so a malformed episode is not expected.

Memory is unused (`{}` is returned); all estimates are rebuilt from the request each call.

## Setup and evaluation commands

Python 3.10+ (developed on 3.12.3; container uses `python:3.12.7-slim`). No third-party packages.

```bash
python -m unittest -v
python verify_data.py
python evaluate.py --seeds 101,102,103 --variants all --output results/final.json   # default policy is 'sm'
./run_all.sh          # all four methods on seeds 101-103, writes results/final_{sm,greedy,no_asks,random}.json
docker build -t sequential-policy:submission .
python evaluate.py --image sequential-policy:submission --seeds 101 --output results/container_check.json
```

`--baseline {sm,greedy,no_asks,random}` selects the method in local mode. The container entrypoint runs `sm` with no arguments.

## Seeds and training

- **Training seeds:** none. There is no trained model; the Beta priors are fixed constants chosen by hand (response 0.8, acceptance 0.4, second-meeting 0.53, strength 3 to 4 pseudo-observations).
- **Inference random seed:** none; the policy is deterministic (a test checks this).
- Public development seeds used for experiments: 101 to 103 (official evaluator), 201, 301 to 304, 401 to 408, 501 to 503 and 601 to 606 (analysis scripts).

## Files

| File | Purpose |
|---|---|
| `sm_policy.py` | The policy |
| `policy.py` | Starter entrypoint, now defaulting to `sm` |
| `test_sm_policy.py` | Empty-state, matching-optimality, episode validity, fallback and determinism tests |
| `fast_eval.py`, `ablate.py`, `proto.py` | Round 1 in-process experiment scripts (prototype, not the submitted policy) |
| `run_all.sh` | Reproduces the four official result files |

## Provenance

- No external data, models or services. Dependencies: Python standard library only.
- Coding assistance: Claude (Anthropic) was used to write and review code and analysis; all reported numbers come from runs of this code.
- The simulator source was read to understand the interface and design experiments. The policy reads only the request state and never simulator objects, seeds or outcomes.

## Limitations

- Outcomes are synthetic. Scores here are not evidence about real people.
- With about one MSMI event per episode, differences between methods are mostly within noise; see the report.
- Per-member propensities barely move from their priors within one 60-day episode.
- Docker was not available in the development environment, so the container build and run have not been executed there. Run the two Docker commands above before submitting.
