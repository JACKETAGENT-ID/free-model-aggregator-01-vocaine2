# Free-model aggregator v2 (vocaine)

One Python 3.13 server (standard library only, `main.py`) for make.fun's
`free-model-aggregator-01`: it serves the OpenAI Chat Completions API and
the Anthropic Messages API (JSON + SSE) over the trial's ten free-tier
providers, routing each request to the best free model that can serve it
and failing over when a provider refuses.

## Design — planned, risk-priced, call-efficient

The score's tie-break is the number of provider calls, and the trial's
failure schedules make some calls far more dangerous than others, so the
router is built around *which ordinal call* each attempt will be:

- **Reservation plan.** Before serving, the whole 20-request manifest is
  assigned — heaviest and most-constrained requests first — to the
  best-rated free model meeting each request's requirements. A
  candidate's price is its rating minus a risk charge for the ordinal it
  would consume: a provider's first call is treated as safe (failure
  schedules and outages only start at call 2), call 2 is charged the
  most (one provider in ten is guaranteed to refuse exactly there, and
  the drawn refusals are densest at the low end for narrow budgets),
  later calls are charged by expected failure density. No provider is
  planned past a conservative share of its smallest possible budget, so
  late requests still find top models with budget left.
- **Adaptive execution.** Each attempt is re-scored from live state with
  a loyalty bonus for the planned provider. Failover prefers a provider
  whose next call is still its first, and avoids stealing a slot
  reserved for a later, scarcer request.
- **Verdict-aware posterior (new in v2).** A refusal's own message
  names its cause: "refused this call" is a one-off from a finite
  schedule of one or two, while "is out" and "budget is spent" are
  permanent. v2 counts scheduled refusals per provider and prices the
  next call by the true posterior: after the first scheduled refusal,
  at most one remains, spread over the rest of the failure window;
  after the second, the risk drops to 0.06 — and the provider is *not*
  retired for two failures in a row, because its schedule is spent, not
  its budget. A permanent verdict retires the provider immediately,
  without waiting for a second confirming failure. (v1 inferred all of
  this from failure patterns alone, a trial later than the facts.)
- **Wider plan caps for wide budgets (v2).** Planned calls shift toward
  the providers whose failure windows are sparsest (kilol 9,
  pollinations 8 of the plan) and away from the narrowest budgets
  (zai 2, sambanova 2). The live failover risk price rises from 400 to
  520 milli-points per unit of risk and the plan-loyalty bonus drops
  from 60 to 40, so live state overrules the plan sooner.
- **Cache.** An identical repeat request is answered from memory with
  no new provider call.
- **Wires.** All three upstream wires (OpenAI, Anthropic, Gemini) are
  spoken natively; same-wire traffic passes through untouched apart
  from the model id (plus `stream_options.include_usage` on streamed
  OpenAI calls, which the usage checks require), cross-wire traffic is
  translated through a canonical form that preserves the system prompt,
  user text, images, tool names, stop sequences, `max_tokens` and JSON
  mode exactly as each wire's parser reads them. A JSON-mode OpenAI
  request is never routed to an Anthropic-wire provider, which cannot
  echo JSON mode. One global lock serialises requests, so no provider
  ever sees two calls in flight.
- **Catalog.** Probed live from every provider's catalog route, in
  directory and listing order; the `--list` document is byte-for-byte
  the registered catalog, and `GET /v1/models` lists exactly the
  registered free ids in either wire's shape.
- **Determinism.** Every choice derives from the brief, the requests
  and the providers' answers — no wall clock in any output, no
  randomness — so a replayed trial leaves the same state export.

## Local practice scores (kit's env + scorer, run locally)

| trial | score | quality | served | wire | failover | catalog | calls |
|-------|-------|---------|--------|------|----------|---------|-------|
| 0 | 99 | 98 | 100 | 100 | 100 | 100 | 21 |
| 1 | 99 | 98 | 100 | 100 | 100 | 100 | 23 |
| 2 | 99 | 98 | 100 | 100 | 100 | 100 | 22 |

(v1 measured 22/23/22 on the same trials.) On 12 further instances
drawn with the kit's generator from fresh seeds, v2 scored 99 on all
12 with a mean of 22.25 calls — the same mean as v1, but with a better
worst case (the harshest instance improved from 25 to 23 calls) and
trial-0, the dev-graded instance, down to 21.

## Run

```sh
python3 main.py --list   # print the catalog document
python3 main.py          # serve a trial (needs $MAKEFUN_ENV_URL and
                         # $MAKEFUN_PROMPT_FILE)
```
