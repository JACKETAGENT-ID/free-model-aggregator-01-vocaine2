#!/usr/bin/env python3
"""Aggregator of free models for make.fun free-model-aggregator-01,
entered by the vocaine account.

A single Python 3.13 program on the standard library alone.  It
answers the OpenAI Chat Completions API and the Anthropic Messages
API, in plain JSON and in SSE, and sends every request to a free model
served by one of the ten trial providers.  All three upstream wires --
OpenAI, Anthropic and Gemini -- are spoken natively, and a request is
translated whenever the client's wire and the provider's wire differ.

How the routing spends as few provider calls as possible:

* Before serving, a plan reserves one primary provider for each
  manifest request, heaviest and most constrained first.  A reservation
  is priced by a risk model of the providers' failure schedules: the
  first call to a provider inside a trial counts as safe, later calls
  are charged by how dense that provider's failure window is expected
  to be, and no provider is planned past a conservative share of the
  smallest budget it could have drawn.
* Serving follows the plan, with a small loyalty bonus for doing so,
  but every attempt is re-scored from live state.  A provider that has
  just refused carries a posterior risk for its next call -- low when a
  wide budget makes a spent one-off refusal likely, high when a narrow
  budget suggests an outage or a spent budget.  Two failures in a row
  retire a provider, and a failure beyond its smallest possible budget
  retires it immediately.
* When a request fails over, providers whose next call would still be
  their first -- a guaranteed slot -- are strongly preferred over
  re-rolling a risky one, and taking a slot the plan reserved for a
  later, scarcer request is penalised.
* A repeat of an identical request is answered from an in-memory cache
  and costs no provider call at all.

The program is a deterministic function of the brief, the requests and
the providers' answers: no output reads a wall clock and nothing is
left to chance.
"""

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_ATTEMPTS = 8

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# Per-provider budget priors, from the competition kit's published
# generator: a trial budget is drawn uniformly from [calls_min,
# calls_max], one or two failure calls are drawn from [2, budget - 1],
# one provider is guaranteed a failure on its second call and one
# provider has an outage starting somewhere in [2, budget - 1].
CALLS_MIN = {"groq": 9, "cerebras": 5, "gemini": 6, "zai": 4,
             "openrouter": 10, "sambanova": 4, "pollinations": 10,
             "cloudflare": 7, "ovh": 5, "kilol": 12}
CALLS_MAX = {"groq": 12, "cerebras": 6, "gemini": 8, "zai": 5,
             "openrouter": 12, "sambanova": 5, "pollinations": 14,
             "cloudflare": 8, "ovh": 6, "kilol": 16}
PLAN_CAP = {"groq": 7, "cerebras": 3, "gemini": 4, "zai": 2,
            "openrouter": 8, "sambanova": 2, "pollinations": 8,
            "cloudflare": 5, "ovh": 3, "kilol": 9}
DEFAULT_MIN, DEFAULT_MAX, DEFAULT_CAP = 6, 10, 4

T_PLAN = 400.0      # milli-points charged per unit of planned risk
T_FAILOVER = 520.0  # milli-points charged per unit of live risk
LOYALTY = 40.0      # bonus for following the plan on the first attempt
STEAL_PEN = 120.0   # penalty for taking a scarcer later request's slot


def calls_min(pid):
    return CALLS_MIN.get(pid, DEFAULT_MIN)


def calls_max(pid):
    return CALLS_MAX.get(pid, DEFAULT_MAX)


def plan_cap(pid):
    return PLAN_CAP.get(pid, DEFAULT_CAP)


def budget_expect(pid):
    return (calls_min(pid) + calls_max(pid)) / 2.0


def prior_risk(pid, n):
    """Estimated probability that provider pid's n-th call refuses,
    before anything has been observed about it this trial."""
    if n <= 1:
        return 0.0
    span = budget_expect(pid) - 2.0
    if n == 2:
        # One provider in ten is guaranteed to fail exactly here, the
        # drawn failures are densest at the low end, and an outage may
        # start here.
        return min(0.9, 0.10 + 0.9 * min(0.85, 1.5 / span) + 0.1 / span)
    risk = 1.2 / span
    if n > calls_min(pid):
        risk = max(risk, 0.5)
    return min(risk, 0.6)


def after_failure_risk(pid):
    """Estimated probability the next call refuses, given the previous
    call just did: a wide-budget provider most likely spent a one-off
    scheduled refusal; a narrow-budget provider may be out or down."""
    b = budget_expect(pid)
    if b >= 10:
        return 0.14
    if b >= 8:
        return 0.20
    if b >= 6.5:
        return 0.30
    if b >= 5:
        return 0.45
    return 0.62


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


# ---------------------------------------------------------------- brief

def _is_record(v):
    return isinstance(v, dict)


def _documents_in(text):
    """Every JSON object embedded in the brief that carries a string
    "schema" field, in order of appearance.  Each "{" is tried as a
    decode start, so prose and nesting cannot hide a document."""
    found = []
    decoder = json.JSONDecoder()
    position = text.find("{")
    while position != -1:
        try:
            value, _end = decoder.raw_decode(text, position)
        except ValueError:
            value = None
        if _is_record(value) and isinstance(value.get("schema"), str):
            found.append(value)
        position = text.find("{", position + 1)
    return found


def _brief_provider(raw):
    # One provider-directory row, with the directory's defaults filled.
    if not _is_record(raw) or not raw.get("id"):
        return None
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if raw.get(key):
            base = raw[key]
            break
    return {
        "id": raw["id"],
        "name": raw.get("name") or raw["id"],
        "wire": raw.get("wire") or "openai",
        "auth": raw.get("auth") or "bearer",
        "base": base,
        "key_env": raw.get("key_env") or "",
        "key": raw.get("key") or "",
    }


def _brief_rating(raw):
    # One quality-table row; ratings stay a list, as the brief gives
    # them, because lookups scan for the first matching row.
    if not _is_record(raw):
        return None
    return {
        "model": raw.get("model"),
        "provider": raw.get("provider"),
        "rating": raw.get("rating_milli") or 0,
        "context_length": raw.get("context_length"),
        "tools": raw.get("tools") is True,
        "vision": raw.get("vision") is True,
        "json": raw.get("json") is True,
        "free": raw.get("free") is True,
    }


def _brief_workload(raw):
    # One workload-manifest row with its requirements normalised.
    if not _is_record(raw) or not raw.get("id"):
        return None
    reqs = raw.get("requires") or {}
    minimum = reqs.get("min_context")
    return {
        "id": raw["id"],
        "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
        "stream": raw.get("stream") is True,
        "weight": 2 if raw.get("weight") == 2 else 1,
        "requires": {
            "tools": reqs.get("tools") is True,
            "vision": reqs.get("vision") is True,
            "json": reqs.get("json") is True,
            "min_context": minimum if isinstance(minimum, (int, float)) else None,
        },
    }


def read_brief(path):
    brief = {"providers": [], "ratings": [], "best": {}, "manifest": []}
    if not path:
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as err:
        log("cannot read brief: %s" % err)
        return brief
    for doc in _documents_in(text):
        schema = doc.get("schema")
        if schema == PROVIDERS_SCHEMA:
            for raw in doc.get("providers", []):
                entry = _brief_provider(raw)
                if entry is not None:
                    brief["providers"].append(entry)
        elif schema == RATINGS_SCHEMA:
            for raw in doc.get("ratings", []):
                entry = _brief_rating(raw)
                if entry is not None:
                    brief["ratings"].append(entry)
            for raw in doc.get("best", []):
                if _is_record(raw):
                    brief["best"][raw.get("id")] = raw.get("rating_milli") or 0
        elif schema == WORKLOAD_SCHEMA:
            for raw in doc.get("manifest", []):
                entry = _brief_workload(raw)
                if entry is not None:
                    brief["manifest"].append(entry)
    return brief


# ---------------------------------------------------------------- probes

def join_base(env_url, base):
    # An absolute base stands alone; a relative one is appended to the
    # environment URL with exactly one joining slash.
    base = (base or "").strip()
    if base.startswith("http://") or base.startswith("https://"):
        return base.rstrip("/")
    if base == "":
        return env_url
    return env_url + ("" if base.startswith("/") else "/") + base.rstrip("/")


_AUTH_SCHEMES = ("bearer", "x-api-key", "x-goog-api-key")


def auth_headers(provider):
    # The single credential header the provider's auth scheme asks
    # for; no key means no header at all.
    key = provider["key"]
    scheme = provider["auth"]
    if key == "" or scheme not in _AUTH_SCHEMES:
        return {}
    if scheme == "bearer":
        return {"authorization": "Bearer " + key}
    return {scheme: key}


def http_call(method, url, headers, payload, timeout=10.0):
    """One HTTP call; returns (status, content type, body bytes).
    An HTTP error status is a result, not an exception."""
    data = None
    hdrs = dict(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, resp.headers.get("content-type", ""), resp.read()
    except urllib.error.HTTPError as err:
        ctype = ""
        if err.headers:
            ctype = err.headers.get("content-type", "")
        return err.code, ctype, err.read()


def _listing_model(wire, raw):
    # One catalog row, normalised across wires: Gemini names its
    # models "models/<id>" and reports inputTokenLimit; the others
    # use "id" and context_window.
    if not _is_record(raw):
        return None
    if wire == "gemini":
        name = raw.get("name")
        mid = name.replace("models/", "", 1) if name else raw.get("id")
        ctx = raw.get("context_length")
        if ctx is None:
            ctx = raw.get("inputTokenLimit")
        display = raw.get("displayName") or raw.get("display_name") or mid
    else:
        mid = raw.get("id")
        ctx = raw.get("context_length")
        if ctx is None:
            ctx = raw.get("context_window")
        display = raw.get("display_name") or raw.get("displayName") or mid
    if not mid:
        return None
    return {
        "id": mid,
        "display_name": display,
        "free": raw.get("free"),
        "context_length": ctx,
        "tools": raw.get("tools"),
        "vision": raw.get("vision"),
        "json": raw.get("json"),
    }


def probe_provider(provider):
    # Catalog routes are free: probing never counts as a provider
    # call.  A failed probe excludes the provider, nothing more.
    base = join_base(ENV_URL, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        status, _ctype, raw_body = http_call("GET", url, headers, None, timeout=5.0)
        if status != 200:
            return {"provider": provider, "ok": False, "models": []}
        body = json.loads(raw_body.decode("utf-8"))
    except Exception as err:
        log("probe %s failed: %s" % (provider["id"], err))
        return {"provider": provider, "ok": False, "models": []}
    items = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    models = []
    if isinstance(items, list):
        for raw_model in items:
            model = _listing_model(provider["wire"], raw_model)
            if model is not None:
                models.append(model)
    return {"provider": provider, "ok": True, "models": models}


def rating_row(brief, pid, mid):
    for row in brief["ratings"]:
        if row["provider"] == pid and row["model"] == mid:
            return row
    return None


def build_catalog(brief, probes):
    """The catalog document, in directory order and listing order.
    Where the quality table has a row for a model it decides the free
    flag, and its context length wins unless it has none."""
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        provider = probe["provider"]
        seen = set()
        models = []
        for model in probe["models"]:
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_row(brief, provider["id"], model["id"])
            if row is not None:
                free = row["free"]
                if row["context_length"] is not None:
                    ctx = row["context_length"]
                else:
                    ctx = model["context_length"]
            else:
                free = model["free"] is True
                ctx = model["context_length"]
            models.append({"id": model["id"], "free": free, "context_length": ctx})
        providers.append({
            "id": provider["id"],
            "base_url": provider["base"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": models,
        })
    return {"providers": providers}


def build_candidates(brief, probes):
    """Routable (provider, model) pairs: listed by the provider and
    free.  Capabilities and rating come from the quality table where
    it has a row; a listing-only model routes with rating 0 and its
    own claims."""
    candidates = []
    for probe in probes:
        if not probe["ok"]:
            continue
        provider = probe["provider"]
        for model in probe["models"]:
            row = rating_row(brief, provider["id"], model["id"])
            if row is not None:
                if not row["free"]:
                    continue
                if row["context_length"] is not None:
                    ctx = row["context_length"]
                else:
                    ctx = model["context_length"]
                candidates.append({
                    "provider": provider,
                    "model": model["id"],
                    "rating": row["rating"],
                    "context_length": ctx,
                    "tools": row["tools"],
                    "vision": row["vision"],
                    "json": row["json"],
                })
            elif model["free"] is True:
                candidates.append({
                    "provider": provider,
                    "model": model["id"],
                    "rating": 0,
                    "context_length": model["context_length"],
                    "tools": model["tools"] is True,
                    "vision": model["vision"] is True,
                    "json": model["json"] is True,
                })
    return candidates


def meets(cand, req):
    if req.get("tools") and not cand["tools"]:
        return False
    if req.get("vision") and not cand["vision"]:
        return False
    if req.get("json") and not cand["json"]:
        return False
    need = req.get("min_context")
    if need is not None:
        if cand["context_length"] is None or cand["context_length"] < need:
            return False
    return True


def best_candidate(candidates, pid, req, api, json_on_openai_wire):
    """A provider's best model for one request, or None."""
    best = None
    for cand in candidates:
        if cand["provider"]["id"] != pid:
            continue
        if not meets(cand, req):
            continue
        if api == "openai" and json_on_openai_wire and cand["provider"]["wire"] == "anthropic":
            continue  # the anthropic wire cannot echo a JSON-mode request
        if best is None or cand["rating"] > best["rating"] or \
                (cand["rating"] == best["rating"] and cand["model"] < best["model"]):
            best = cand
    return best


def eligible_providers(candidates, provider_ids, req, api, json_on_openai_wire):
    return [pid for pid in provider_ids
            if best_candidate(candidates, pid, req, api, json_on_openai_wire) is not None]


def make_plan(brief, candidates, provider_ids):
    """Reserve one primary provider per manifest request: heaviest and
    most-constrained first, risk-priced, under per-provider plan caps."""
    manifest = brief["manifest"]
    planned = {pid: 0 for pid in provider_ids}
    plan = {}
    order = sorted(range(len(manifest)),
                   key=lambda i: (-manifest[i]["weight"],
                                  len(eligible_providers(
                                      candidates, provider_ids, manifest[i]["requires"],
                                      manifest[i]["api"], manifest[i]["requires"]["json"])),
                                  i))
    for i in order:
        entry = manifest[i]
        req = entry["requires"]
        json_wire = req["json"]
        best = None
        for pid in provider_ids:
            if planned[pid] >= plan_cap(pid):
                continue
            cand = best_candidate(candidates, pid, req, entry["api"], json_wire)
            if cand is None:
                continue
            eff = cand["rating"] - T_PLAN * prior_risk(pid, planned[pid] + 1)
            key = (eff, cand["rating"], pid)
            if best is None or key > best[0]:
                best = (key, pid)
        if best is not None:
            plan[entry["id"]] = best[1]
            planned[best[1]] += 1
    return plan


# ------------------------------------------------------- canonical form
# A client request, whichever wire it arrived on:
#   system: str; messages: [{role, blocks}]; tools: [{name, description,
#   parameters}]; tool_choice: None | "auto" | "required" | {"name": str};
#   max_tokens / temperature / top_p: value or None; stops: [str];
#   json_mode: bool; stream: bool.
# Blocks: {"kind": "text", "text"} | {"kind": "image", "media_type", "data"}
#       | {"kind": "image_url", "url"}
#       | {"kind": "tool_use", "id", "name", "input"}
#       | {"kind": "tool_result", "tool_use_id", "text", "is_error"}

def _blocks_text(content):
    # The plain text of a content value: the string itself, or the
    # text blocks inside a block list joined by newlines.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for block in content:
            if _is_record(block) and block.get("type") == "text" \
                    and isinstance(block.get("text"), str):
                pieces.append(block["text"])
        return "\n".join(pieces)
    return ""


def _image_block_from_data_url(url):
    # data:<media>;base64,<data> becomes an inline image block; a URL
    # that does not split stays a plain image URL.
    if "," not in url:
        return {"kind": "image_url", "url": url}
    head, data = url.split(",", 1)
    media = head[5:].split(";")[0]
    if media == "":
        media = "image/png"
    return {"kind": "image", "media_type": media, "data": data}


def _arguments_value(raw_args):
    # A tool call's arguments: a JSON string that decodes to an
    # object, an object as-is, anything else an empty object.
    if isinstance(raw_args, str) and raw_args.strip() != "":
        try:
            value = json.loads(raw_args)
        except ValueError:
            return {}
        return value if _is_record(value) else {}
    if _is_record(raw_args):
        return raw_args
    return {}


def _openai_content_blocks(content):
    # The canonical blocks of an OpenAI message content value.
    blocks = []
    if isinstance(content, str):
        if content != "":
            blocks.append({"kind": "text", "text": content})
        return blocks
    if not isinstance(content, list):
        return blocks
    for part in content:
        if not _is_record(part):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            blocks.append({"kind": "text", "text": part["text"]})
        elif part.get("type") == "image_url" and _is_record(part.get("image_url")):
            url = part["image_url"].get("url") or ""
            if url.startswith("data:"):
                blocks.append(_image_block_from_data_url(url))
            elif url != "":
                blocks.append({"kind": "image_url", "url": url})
    return blocks


def _openai_tool_blocks(raw):
    # The canonical tool_use blocks of an OpenAI assistant message.
    blocks = []
    for call in raw.get("tool_calls") or []:
        if not _is_record(call) or not _is_record(call.get("function")):
            continue
        function = call["function"]
        blocks.append({
            "kind": "tool_use",
            "id": call.get("id") or "",
            "name": function.get("name") or "",
            "input": _arguments_value(function.get("arguments")),
        })
    return blocks


def parse_openai_request(body):
    # An OpenAI chat request in canonical form.  A tool message is a
    # user message holding a single tool_result block.
    system_parts = []
    messages = []
    for raw in body.get("messages") or []:
        if not _is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = _blocks_text(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "tool":
            messages.append({"role": "user", "blocks": [{
                "kind": "tool_result",
                "tool_use_id": raw.get("tool_call_id") or "",
                "text": _blocks_text(content),
                "is_error": False,
            }]})
            continue
        blocks = _openai_content_blocks(content)
        if role == "assistant":
            blocks = blocks + _openai_tool_blocks(raw)
            messages.append({"role": "assistant", "blocks": blocks})
        else:
            messages.append({"role": "user", "blocks": blocks})
    tools = []
    for raw in body.get("tools") or []:
        if _is_record(raw) and _is_record(raw.get("function")) and raw["function"].get("name"):
            function = raw["function"]
            parameters = function.get("parameters")
            tools.append({
                "name": function["name"],
                "description": function.get("description") or "",
                "parameters": parameters if _is_record(parameters)
                else {"type": "object", "properties": {}},
            })
    choice = body.get("tool_choice")
    tool_choice = None
    if choice == "auto":
        tool_choice = "auto"
    elif choice == "required":
        tool_choice = "required"
    elif _is_record(choice) and _is_record(choice.get("function")) and choice["function"].get("name"):
        tool_choice = {"name": choice["function"]["name"]}
    stops = body.get("stop")
    if isinstance(stops, str):
        stops = [stops]
    elif isinstance(stops, list):
        stops = [s for s in stops if isinstance(s, str)]
    else:
        stops = []
    fmt = body.get("response_format")
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    return {
        "system": "\n".join(system_parts),
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": max_tokens if isinstance(max_tokens, (int, float)) else None,
        "temperature": body.get("temperature") if isinstance(body.get("temperature"), (int, float)) else None,
        "top_p": body.get("top_p") if isinstance(body.get("top_p"), (int, float)) else None,
        "stops": stops,
        "json_mode": _is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"),
        "stream": body.get("stream") is True,
    }


def _anthropic_content_blocks(content):
    # The canonical blocks of an Anthropic message content value.
    blocks = []
    if isinstance(content, str):
        if content != "":
            blocks.append({"kind": "text", "text": content})
        return blocks
    if not isinstance(content, list):
        return blocks
    for part in content:
        if not _is_record(part):
            continue
        kind = part.get("type")
        if kind == "text" and isinstance(part.get("text"), str):
            blocks.append({"kind": "text", "text": part["text"]})
        elif kind == "image" and _is_record(part.get("source")):
            source = part["source"]
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                blocks.append({
                    "kind": "image",
                    "media_type": source.get("media_type") or "image/png",
                    "data": source["data"],
                })
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                blocks.append({"kind": "image_url", "url": source["url"]})
        elif kind == "tool_use":
            given = part.get("input")
            blocks.append({
                "kind": "tool_use",
                "id": part.get("id") or "",
                "name": part.get("name") or "",
                "input": given if _is_record(given) else {},
            })
        elif kind == "tool_result":
            inner = part.get("content")
            text = inner if isinstance(inner, str) else _blocks_text(inner)
            blocks.append({
                "kind": "tool_result",
                "tool_use_id": part.get("tool_use_id") or "",
                "text": text,
                "is_error": part.get("is_error") is True,
            })
    return blocks


def parse_anthropic_request(body):
    # An Anthropic Messages request in canonical form; json_mode is
    # always False because the wire has no JSON mode to echo.
    messages = []
    for raw in body.get("messages") or []:
        if not _is_record(raw):
            continue
        role = "assistant" if raw.get("role") == "assistant" else "user"
        messages.append({"role": role, "blocks": _anthropic_content_blocks(raw.get("content"))})
    tools = []
    for raw in body.get("tools") or []:
        if _is_record(raw) and raw.get("name"):
            schema = raw.get("input_schema")
            tools.append({
                "name": raw["name"],
                "description": raw.get("description") or "",
                "parameters": schema if _is_record(schema)
                else {"type": "object", "properties": {}},
            })
    choice = body.get("tool_choice")
    tool_choice = None
    if _is_record(choice):
        if choice.get("type") == "auto":
            tool_choice = "auto"
        elif choice.get("type") == "any":
            tool_choice = "required"
        elif choice.get("type") == "tool" and choice.get("name"):
            tool_choice = {"name": choice["name"]}
    stops = body.get("stop_sequences")
    if isinstance(stops, list):
        stops = [s for s in stops if isinstance(s, str)]
    else:
        stops = []
    return {
        "system": _blocks_text(body.get("system")),
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": body.get("max_tokens") if isinstance(body.get("max_tokens"), (int, float)) else None,
        "temperature": body.get("temperature") if isinstance(body.get("temperature"), (int, float)) else None,
        "top_p": body.get("top_p") if isinstance(body.get("top_p"), (int, float)) else None,
        "stops": stops,
        "json_mode": False,
        "stream": body.get("stream") is True,
    }


def _openai_assistant_message(blocks):
    # An assistant message: joined text (null when there is none) and
    # one function call per tool_use block, ids defaulted by position.
    text = "\n".join(b["text"] for b in blocks if b["kind"] == "text")
    calls = []
    for i, block in enumerate(blocks):
        if block["kind"] != "tool_use":
            continue
        calls.append({
            "id": block["id"] or "call_%d" % i,
            "type": "function",
            "function": {"name": block["name"], "arguments": json.dumps(block["input"])},
        })
    message = {"role": "assistant", "content": text if text != "" else None}
    if calls:
        message["tool_calls"] = calls
    return message


def _openai_user_part(block):
    if block["kind"] == "text":
        return {"type": "text", "text": block["text"]}
    if block["kind"] == "image":
        return {"type": "image_url", "image_url": {
            "url": "data:%s;base64,%s" % (block["media_type"], block["data"])}}
    if block["kind"] == "image_url":
        return {"type": "image_url", "image_url": {"url": block["url"]}}
    return None


def render_openai_request(canon, model):
    # Canonical form -> an OpenAI chat request.  Inside a user
    # message, ordinary blocks accumulate until a tool_result flushes
    # them and becomes a tool message of its own; a lone text block
    # renders as a plain string.
    messages = []
    if canon["system"] != "":
        messages.append({"role": "system", "content": canon["system"]})
    for msg in canon["messages"]:
        if msg["role"] == "assistant":
            messages.append(_openai_assistant_message(msg["blocks"]))
            continue
        pending = []

        def flush_pending():
            if not pending:
                return
            if len(pending) == 1 and pending[0]["kind"] == "text":
                messages.append({"role": "user", "content": pending[0]["text"]})
            else:
                parts = []
                for block in pending:
                    part = _openai_user_part(block)
                    if part is not None:
                        parts.append(part)
                if parts:
                    messages.append({"role": "user", "content": parts})
            pending.clear()

        for block in msg["blocks"]:
            if block["kind"] == "tool_result":
                flush_pending()
                messages.append({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": block["text"],
                })
            else:
                pending.append(block)
        flush_pending()
    out = {"model": model, "messages": messages}
    if canon["max_tokens"] is not None:
        out["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        out["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        out["top_p"] = canon["top_p"]
    if canon["stops"]:
        out["stop"] = canon["stops"]
    if canon["tools"]:
        out["tools"] = [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        } for tool in canon["tools"]]
    if canon["tool_choice"] == "auto":
        out["tool_choice"] = "auto"
    elif canon["tool_choice"] == "required":
        out["tool_choice"] = "required"
    elif _is_record(canon["tool_choice"]):
        out["tool_choice"] = {
            "type": "function",
            "function": {"name": canon["tool_choice"]["name"]},
        }
    if canon["json_mode"]:
        out["response_format"] = {"type": "json_object"}
    if canon["stream"]:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _anthropic_block_of(block):
    # One canonical block on the Anthropic wire.
    kind = block["kind"]
    if kind == "text":
        return {"type": "text", "text": block["text"]}
    if kind == "image":
        return {"type": "image", "source": {
            "type": "base64", "media_type": block["media_type"], "data": block["data"]}}
    if kind == "image_url":
        return {"type": "image", "source": {"type": "url", "url": block["url"]}}
    if kind == "tool_use":
        return {"type": "tool_use", "id": block["id"] or "toolu_0",
                "name": block["name"], "input": block["input"]}
    if kind == "tool_result":
        return {"type": "tool_result", "tool_use_id": block["tool_use_id"],
                "content": block["text"], "is_error": block["is_error"]}
    return None


def render_anthropic_request(canon, model):
    # Canonical form -> an Anthropic Messages request.
    messages = []
    for msg in canon["messages"]:
        blocks = []
        for block in msg["blocks"]:
            rendered = _anthropic_block_of(block)
            if rendered is not None:
                blocks.append(rendered)
        messages.append({"role": msg["role"], "content": blocks})
    out = {"model": model, "messages": messages}
    if canon["system"] != "":
        out["system"] = canon["system"]
    if canon["max_tokens"] is not None:
        out["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        out["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        out["top_p"] = canon["top_p"]
    if canon["stops"]:
        out["stop_sequences"] = canon["stops"]
    if canon["tools"]:
        out["tools"] = [{
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        } for tool in canon["tools"]]
    if canon["tool_choice"] == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif canon["tool_choice"] == "required":
        out["tool_choice"] = {"type": "any"}
    elif _is_record(canon["tool_choice"]):
        out["tool_choice"] = {"type": "tool", "name": canon["tool_choice"]["name"]}
    if canon["stream"]:
        out["stream"] = True
    return out


def _gemini_part_of(block):
    # One canonical block as a Gemini part; a tool result answers
    # under its tool_use_id, or "tool" when it has none.
    kind = block["kind"]
    if kind == "text":
        return {"text": block["text"]}
    if kind == "image":
        return {"inlineData": {"mimeType": block["media_type"], "data": block["data"]}}
    if kind == "image_url":
        return {"fileData": {"fileUri": block["url"]}}
    if kind == "tool_use":
        return {"functionCall": {"name": block["name"], "args": block["input"]}}
    if kind == "tool_result":
        return {"functionResponse": {
            "name": block["tool_use_id"] or "tool",
            "response": {"result": block["text"]}}}
    return None


def render_gemini_request(canon, model):
    # Canonical form -> a Gemini generateContent request.
    contents = []
    for msg in canon["messages"]:
        parts = []
        for block in msg["blocks"]:
            part = _gemini_part_of(block)
            if part is not None:
                parts.append(part)
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": parts})
    out = {"contents": contents}
    if canon["system"] != "":
        out["systemInstruction"] = {"parts": [{"text": canon["system"]}]}
    if canon["tools"]:
        declarations = [{
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        } for tool in canon["tools"]]
        out["tools"] = [{"functionDeclarations": declarations}]
    config = {}
    if canon["max_tokens"] is not None:
        config["maxOutputTokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        config["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        config["topP"] = canon["top_p"]
    if canon["stops"]:
        config["stopSequences"] = canon["stops"]
    if canon["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


# ---------------------------------------------------- response handling
# A provider's answer in canonical form:
#   text: str; tool_calls: [{id, name, arguments}]; family: stop family
#   (0 end, 1 length, 2 tool use, 3 refusal); usage_in / usage_out: int or
#   None; resp_id: the provider's own id for the answer, or None.

STOP_FAMILY = {
    "stop": 0, "end_turn": 0, "STOP": 0, "stop_sequence": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_FINISH = ["stop", "length", "tool_calls", "content_filter"]
ANTHROPIC_STOP = ["end_turn", "max_tokens", "tool_use", "refusal"]


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _sse_frames(raw):
    # Split a raw SSE body into (event, data) pairs.  A data line's
    # value loses its leading spaces only when it starts with one.
    text = raw.decode("utf-8", "replace").replace("\r\n", "\n")
    frames = []
    for frame in text.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data_lines = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                rest = line[len("data:"):]
                if rest.startswith(" "):
                    rest = rest.lstrip(" ")
                data_lines.append(rest)
        if data_lines:
            frames.append((event, "\n".join(data_lines)))
    return frames


def _empty_answer(model):
    return {"text": "", "tool_calls": [], "family": 0,
            "usage_in": None, "usage_out": None, "resp_id": None, "model": model}


def _openai_answer_json(ans, raw):
    try:
        body = json.loads(raw.decode("utf-8"))
    except ValueError:
        return ans
    choices = body.get("choices") or []
    choice = choices[0] if choices and _is_record(choices[0]) else {}
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        ans["text"] = content
    elif isinstance(content, list):
        ans["text"] = _blocks_text(content)
    for call in message.get("tool_calls") or []:
        if _is_record(call) and _is_record(call.get("function")):
            function = call["function"]
            ans["tool_calls"].append({
                "id": call.get("id") or "",
                "name": function.get("name") or "",
                "arguments": function.get("arguments") or "",
            })
    ans["family"] = STOP_FAMILY.get(choice.get("finish_reason"), 0)
    usage = body.get("usage") or {}
    ans["usage_in"] = _num(usage.get("prompt_tokens"))
    ans["usage_out"] = _num(usage.get("completion_tokens"))
    ans["resp_id"] = body.get("id")
    return ans


def _openai_answer_stream(ans, raw):
    # Chunks accumulate: text concatenates, tool calls assemble per
    # upstream index, usage and finish come from whichever chunk
    # carries them last.
    calls = {}
    for _event, payload in _sse_frames(raw):
        if payload.strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not _is_record(chunk):
            continue
        if _is_record(chunk.get("usage")):
            ans["usage_in"] = _num(chunk["usage"].get("prompt_tokens"))
            ans["usage_out"] = _num(chunk["usage"].get("completion_tokens"))
        if not ans["resp_id"]:
            ans["resp_id"] = chunk.get("id")
        choices = chunk.get("choices") or []
        if not choices or not _is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            ans["text"] += delta["content"]
        for call in delta.get("tool_calls") or []:
            if not _is_record(call):
                continue
            idx = call.get("index") if isinstance(call.get("index"), int) else len(calls)
            slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if call.get("id"):
                slot["id"] = call["id"]
            function = call.get("function") or {}
            if function.get("name"):
                slot["name"] = function["name"]
            if isinstance(function.get("arguments"), str):
                slot["arguments"] += function["arguments"]
        if choice.get("finish_reason") is not None:
            ans["family"] = STOP_FAMILY.get(choice.get("finish_reason"), 0)
    ans["tool_calls"] = [calls[key] for key in sorted(calls)]
    return ans


def parse_openai_answer(raw, stream, model):
    ans = _empty_answer(model)
    if stream:
        return _openai_answer_stream(ans, raw)
    return _openai_answer_json(ans, raw)


def _anthropic_answer_json(ans, raw):
    try:
        body = json.loads(raw.decode("utf-8"))
    except ValueError:
        return ans
    for block in body.get("content") or []:
        if not _is_record(block):
            continue
        if block.get("type") == "text":
            ans["text"] += block.get("text") or ""
        elif block.get("type") == "tool_use":
            ans["tool_calls"].append({
                "id": block.get("id") or "",
                "name": block.get("name") or "",
                "arguments": json.dumps(block.get("input") or {}),
            })
    ans["family"] = STOP_FAMILY.get(body.get("stop_reason"), 0)
    usage = body.get("usage") or {}
    ans["usage_in"] = _num(usage.get("input_tokens"))
    ans["usage_out"] = _num(usage.get("output_tokens"))
    ans["resp_id"] = body.get("id")
    return ans


def _anthropic_answer_stream(ans, raw):
    # Blocks assemble per content index; at the end, tool blocks
    # become calls and the rest concatenates into the text, in index
    # order.
    blocks = {}

    def slot_for(idx, kind):
        return blocks.setdefault(idx, {"kind": kind, "text": "",
                                       "id": "", "name": "", "arguments": ""})

    for event, payload in _sse_frames(raw):
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if not _is_record(data):
            continue
        if event == "message_start":
            message = data.get("message") or {}
            if not ans["resp_id"]:
                ans["resp_id"] = message.get("id")
            usage = message.get("usage") or {}
            ans["usage_in"] = _num(usage.get("input_tokens"))
        elif event == "content_block_start":
            block = data.get("content_block") or {}
            slot = slot_for(data.get("index", 0), block.get("type"))
            if block.get("type") == "tool_use":
                slot["id"] = block.get("id") or ""
                slot["name"] = block.get("name") or ""
        elif event == "content_block_delta":
            slot = slot_for(data.get("index", 0), "text")
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                slot["text"] += delta.get("text") or ""
            elif delta.get("type") == "input_json_delta":
                slot["arguments"] += delta.get("partial_json") or ""
        elif event == "message_delta":
            delta = data.get("delta") or {}
            if delta.get("stop_reason") is not None:
                ans["family"] = STOP_FAMILY.get(delta.get("stop_reason"), 0)
            usage = data.get("usage") or {}
            ans["usage_out"] = _num(usage.get("output_tokens"))
    for idx in sorted(blocks):
        slot = blocks[idx]
        if slot["kind"] == "tool_use":
            ans["tool_calls"].append({
                "id": slot["id"], "name": slot["name"], "arguments": slot["arguments"]})
        else:
            ans["text"] += slot["text"]
    return ans


def parse_anthropic_answer(raw, stream, model):
    ans = _empty_answer(model)
    if stream:
        return _anthropic_answer_stream(ans, raw)
    return _anthropic_answer_json(ans, raw)


def parse_gemini_answer(raw, stream, model):
    ans = _empty_answer(model)

    def absorb(body):
        cands = body.get("candidates") or []
        cand = cands[0] if cands and _is_record(cands[0]) else {}
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            if not _is_record(part):
                continue
            if isinstance(part.get("text"), str):
                ans["text"] += part["text"]
            elif _is_record(part.get("functionCall")):
                fn = part["functionCall"]
                ans["tool_calls"].append({
                    "id": "", "name": fn.get("name") or "",
                    "arguments": json.dumps(fn.get("args") or {})})
        if cand.get("finishReason") is not None:
            ans["family"] = STOP_FAMILY.get(cand.get("finishReason"), 0)
        usage = body.get("usageMetadata") or {}
        if usage:
            ans["usage_in"] = _num(usage.get("promptTokenCount"))
            ans["usage_out"] = _num(usage.get("candidatesTokenCount"))

    if not stream:
        try:
            absorb(json.loads(raw.decode("utf-8")))
        except ValueError:
            pass
        return ans
    for _event, payload in _sse_frames(raw):
        try:
            body = json.loads(payload)
        except ValueError:
            continue
        if _is_record(body):
            absorb(body)
    return ans


def render_openai_answer(ans, model):
    message = {"role": "assistant", "content": ans["text"]}
    if ans["tool_calls"]:
        message["tool_calls"] = [{
            "id": c["id"] or "call_%d" % i, "type": "function",
            "function": {"name": c["name"], "arguments": c["arguments"]}}
            for i, c in enumerate(ans["tool_calls"])]
    usage_in = int(ans["usage_in"]) if ans["usage_in"] is not None else 0
    usage_out = int(ans["usage_out"]) if ans["usage_out"] is not None else 0
    return {"id": "chatcmpl-" + (ans["resp_id"] or "aggregate"),
            "object": "chat.completion", "created": FIXED_CREATED, "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": OPENAI_FINISH[ans["family"]]}],
            "usage": {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                      "total_tokens": usage_in + usage_out}}


def render_anthropic_answer(ans, model):
    content = []
    if ans["text"] or not ans["tool_calls"]:
        content.append({"type": "text", "text": ans["text"]})
    for i, call in enumerate(ans["tool_calls"]):
        try:
            parsed = json.loads(call["arguments"]) if call["arguments"] else {}
        except ValueError:
            parsed = {}
        content.append({"type": "tool_use", "id": call["id"] or "toolu_%d" % i,
                        "name": call["name"],
                        "input": parsed if _is_record(parsed) else {}})
    return {"id": "msg_" + (ans["resp_id"] or "aggregate"), "type": "message",
            "role": "assistant", "model": model, "content": content,
            "stop_reason": ANTHROPIC_STOP[ans["family"]], "stop_sequence": None,
            "usage": {"input_tokens": int(ans["usage_in"]) if ans["usage_in"] is not None else 0,
                      "output_tokens": int(ans["usage_out"]) if ans["usage_out"] is not None else 0}}


def _sse(data, event=None):
    return ((("event: %s\n" % event) if event else "")
            + "data: %s\n\n" % json.dumps(data))


def render_openai_stream(ans, model):
    base = {"id": "chatcmpl-" + (ans["resp_id"] or "aggregate"),
            "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model}
    out = [_sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                      "finish_reason": None}]})]
    if ans["text"]:
        out.append(_sse({**base, "choices": [{"index": 0, "delta": {"content": ans["text"]},
                                              "finish_reason": None}]}))
    for i, call in enumerate(ans["tool_calls"]):
        out.append(_sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": i, "id": call["id"] or "call_%d" % i, "type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"]}}]},
            "finish_reason": None}]}))
    out.append(_sse({**base, "choices": [{"index": 0, "delta": {},
                                          "finish_reason": OPENAI_FINISH[ans["family"]]}]}))
    usage_in = int(ans["usage_in"]) if ans["usage_in"] is not None else 0
    usage_out = int(ans["usage_out"]) if ans["usage_out"] is not None else 0
    out.append(_sse({**base, "choices": [], "usage": {
        "prompt_tokens": usage_in, "completion_tokens": usage_out,
        "total_tokens": usage_in + usage_out}}))
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


def render_anthropic_stream(ans, model):
    usage_in = int(ans["usage_in"]) if ans["usage_in"] is not None else 0
    usage_out = int(ans["usage_out"]) if ans["usage_out"] is not None else 0
    out = [_sse({"type": "message_start", "message": {
        "id": "msg_" + (ans["resp_id"] or "aggregate"), "type": "message",
        "role": "assistant", "model": model, "content": [], "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": usage_in, "output_tokens": 0}}}, "message_start")]
    index = 0
    if ans["text"] or not ans["tool_calls"]:
        out.append(_sse({"type": "content_block_start", "index": index,
                         "content_block": {"type": "text", "text": ""}}, "content_block_start"))
        if ans["text"]:
            out.append(_sse({"type": "content_block_delta", "index": index,
                             "delta": {"type": "text_delta", "text": ans["text"]}},
                            "content_block_delta"))
        out.append(_sse({"type": "content_block_stop", "index": index}, "content_block_stop"))
        index += 1
    for i, call in enumerate(ans["tool_calls"]):
        out.append(_sse({"type": "content_block_start", "index": index, "content_block": {
            "type": "tool_use", "id": call["id"] or "toolu_%d" % i,
            "name": call["name"], "input": {}}}, "content_block_start"))
        if call["arguments"]:
            out.append(_sse({"type": "content_block_delta", "index": index, "delta": {
                "type": "input_json_delta", "partial_json": call["arguments"]}},
                "content_block_delta"))
        out.append(_sse({"type": "content_block_stop", "index": index}, "content_block_stop"))
        index += 1
    out.append(_sse({"type": "message_delta", "delta": {
        "stop_reason": ANTHROPIC_STOP[ans["family"]], "stop_sequence": None},
        "usage": {"output_tokens": usage_out}}, "message_delta"))
    out.append(_sse({"type": "message_stop"}, "message_stop"))
    return "".join(out).encode("utf-8")


# ------------------------------------------------------------- upstream

def provider_url(provider, model, stream):
    base = join_base(ENV_URL, provider["base"])
    if provider["wire"] == "openai":
        return base + "/v1/chat/completions"
    if provider["wire"] == "anthropic":
        return base + "/v1/messages"
    verb = "streamGenerateContent" if stream else "generateContent"
    url = "%s/v1beta/models/%s:%s" % (base, urllib.parse.quote(model, safe=""), verb)
    if stream:
        url += "?alt=sse"
    return url


def call_provider(provider, payload, model, stream):
    headers = {"accept": "text/event-stream" if stream else "application/json"}
    headers.update(auth_headers(provider))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    return http_call("POST", provider_url(provider, model, stream),
                     headers, payload, timeout=25.0)


def error_code_of(wire, raw):
    try:
        body = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None
    err = body.get("error") if _is_record(body) else None
    if not _is_record(err):
        return None
    if wire == "gemini":
        return err.get("status")
    return err.get("code") or err.get("type")


def refusal_kind(raw):
    """The cause a refusal names in its own message: "dead" when the
    provider says its budget is spent or it is out (permanent for this
    trial), "sched" for a one-off scheduled refusal, "" when unnamed."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except ValueError:
        return ""
    err = body.get("error") if _is_record(body) else None
    if not _is_record(err):
        return ""
    message = err.get("message")
    if not isinstance(message, str):
        return ""
    if "budget is spent" in message or "is out" in message:
        return "dead"
    if "refused this call" in message:
        return "sched"
    return ""


def parse_upstream(wire, raw, stream, model):
    if wire == "openai":
        return parse_openai_answer(raw, stream, model)
    if wire == "anthropic":
        return parse_anthropic_answer(raw, stream, model)
    return parse_gemini_answer(raw, stream, model)


# --------------------------------------------------------------- server

class Aggregator:
    def __init__(self, brief, probes, catalog):
        self.brief = brief
        self.probes = probes
        self.catalog = catalog
        self.candidates = build_candidates(brief, probes)
        self.provider_ids = [p["provider"]["id"] for p in probes if p["ok"]]
        self.providers = {p["provider"]["id"]: p["provider"] for p in probes if p["ok"]}
        self.plan = make_plan(brief, self.candidates, self.provider_ids)
        self.state = {pid: {"calls": 0, "consec": 0, "survived2": False, "dead": False,
                            "sched": 0}
                      for pid in self.provider_ids}
        self.manifest = brief["manifest"]
        self.manifest_index = 0
        self.cache = {}
        self.lock = threading.Lock()
        # Later requests' reservations: provider -> smallest eligible
        # count among later manifest entries planned onto it.
        self.elig_counts = []
        for entry in self.manifest:
            self.elig_counts.append(len(eligible_providers(
                self.candidates, self.provider_ids, entry["requires"],
                entry["api"], entry["requires"]["json"])))

    # -- model list ------------------------------------------------------
    def served_models(self):
        out = []
        seen = set()
        display = {}
        for probe in self.probes:
            if not probe["ok"]:
                continue
            for model in probe["models"]:
                display.setdefault(model["id"], model["display_name"])
        for provider in self.catalog["providers"]:
            for model in provider["models"]:
                if model["free"] and model["id"] not in seen:
                    seen.add(model["id"])
                    out.append({"id": model["id"], "provider": provider["id"],
                                "display_name": display.get(model["id"], model["id"])})
        return out

    def models_body(self, anthropic_shape):
        models = self.served_models()
        if anthropic_shape:
            return {"data": [{"type": "model", "id": m["id"],
                              "display_name": m["display_name"],
                              "created_at": FIXED_CREATED_AT} for m in models],
                    "has_more": False}
        return {"object": "list",
                "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                          "owned_by": m["provider"]} for m in models]}

    # -- routing ----------------------------------------------------------
    def requirements_for(self, index, api, body, canon):
        req = {"tools": False, "vision": False, "json": False, "min_context": None}
        entry = self.manifest[index] if 0 <= index < len(self.manifest) else None
        if entry is not None:
            req.update(entry["requires"])
        if canon["tools"]:
            req["tools"] = True
        for msg in canon["messages"]:
            for block in msg["blocks"]:
                if block["kind"] in ("image", "image_url"):
                    req["vision"] = True
        if api == "openai" and canon["json_mode"]:
            req["json"] = True
        return req, entry

    def live_risk(self, pid, st, n):
        if st["sched"] >= 2:
            # Both scheduled refusals are spent (the messages said so):
            # only an outage or the budget wall can still refuse.
            risk = 0.06
        elif st["sched"] == 1:
            # One scheduled refusal spent; at most one remains, spread
            # over the rest of the provider's failure window.
            risk = min(0.35, 0.55 / max(1.0, budget_expect(pid) - st["calls"] - 1.0) + 0.05)
        elif st["consec"] >= 1:
            risk = after_failure_risk(pid)
        elif n == 1:
            risk = 0.0
        elif st["survived2"]:
            risk = prior_risk(pid, n) * 0.6
        else:
            risk = prior_risk(pid, n)
        if n > calls_min(pid):
            risk = max(risk, 0.5)
        return risk

    def pick(self, req, api, json_wire, index, entry, attempts):
        my_elig = self.elig_counts[index] if 0 <= index < len(self.elig_counts) else 99
        later = {}
        for j in range(index + 1, len(self.manifest)):
            pid = self.plan.get(self.manifest[j]["id"])
            if pid:
                later.setdefault(pid, []).append(self.elig_counts[j])
        best = None
        for pid in self.provider_ids:
            st = self.state[pid]
            if st["dead"] or (st["consec"] >= 2 and st["sched"] < 2):
                continue
            cand = best_candidate(self.candidates, pid, req, api, json_wire)
            if cand is None:
                continue
            n = st["calls"] + 1
            eff = cand["rating"] - T_FAILOVER * self.live_risk(pid, st, n)
            if attempts == 0 and entry is not None and self.plan.get(entry["id"]) == pid:
                eff += LOYALTY
            if pid in later and not (entry is not None and self.plan.get(entry["id"]) == pid):
                if min(later[pid]) < my_elig:
                    eff -= STEAL_PEN
            key = (-eff, -cand["rating"], pid, cand["model"])
            if best is None or key < best[0]:
                best = (key, cand)
        return best[1] if best else None

    def serve(self, api, body):
        """Answer one client inference request; returns
        (status, content_type, payload bytes)."""
        with self.lock:
            index = self.manifest_index
            self.manifest_index += 1
            cache_key = hashlib.sha256(
                (api + "\n").encode() + json.dumps(
                    body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if cache_key in self.cache:
                return self.cache[cache_key]
            canon = parse_openai_request(body) if api == "openai" \
                else parse_anthropic_request(body)
            req, entry = self.requirements_for(index, api, body, canon)
            json_wire = canon["json_mode"] if api == "openai" else False
            stream = canon["stream"]
            result = self._serve_fresh(api, body, canon, req, entry, index,
                                       json_wire, stream)
            if result[0] == 200:
                self.cache[cache_key] = result
            return result

    def _serve_fresh(self, api, body, canon, req, entry, index, json_wire, stream):
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            cand = self.pick(req, api, json_wire, index, entry, attempts)
            if cand is None:
                break
            provider = cand["provider"]
            pid = provider["id"]
            st = self.state[pid]
            attempts += 1
            same_wire = provider["wire"] == api
            if same_wire:
                payload = dict(body)
                payload["model"] = cand["model"]
                if provider["wire"] == "openai" and stream:
                    payload["stream"] = True
                    payload["stream_options"] = {"include_usage": True}
            elif provider["wire"] == "openai":
                payload = render_openai_request(canon, cand["model"])
            elif provider["wire"] == "anthropic":
                payload = render_anthropic_request(canon, cand["model"])
            else:
                payload = render_gemini_request(canon, cand["model"])
            try:
                status, ctype, raw = call_provider(provider, payload,
                                                   cand["model"], stream)
            except Exception as err:
                log("provider %s unreachable: %s" % (pid, err))
                st["consec"] += 1
                if st["consec"] >= 2:
                    st["dead"] = True
                continue
            if status == 429 and error_code_of(provider["wire"], raw) == "concurrent_requests":
                # Spent none of the provider's budget: do not count it.
                continue
            st["calls"] += 1
            if 200 <= status < 300:
                st["consec"] = 0
                if st["calls"] >= 2:
                    st["survived2"] = True
                if same_wire:
                    return status, ctype or ("text/event-stream" if stream
                                              else "application/json"), raw
                ans = parse_upstream(provider["wire"], raw, stream, cand["model"])
                if api == "openai":
                    if stream:
                        return 200, "text/event-stream", render_openai_stream(ans, cand["model"])
                    return 200, "application/json", json.dumps(
                        render_openai_answer(ans, cand["model"])).encode()
                if stream:
                    return 200, "text/event-stream", render_anthropic_stream(ans, cand["model"])
                return 200, "application/json", json.dumps(
                    render_anthropic_answer(ans, cand["model"])).encode()
            # A refusal: update what we know about the provider. The
            # refusal's own message names the cause: a spent budget or
            # an outage retires the provider on the spot, while a
            # scheduled one-off is counted against its finite schedule.
            st["consec"] += 1
            kind = refusal_kind(raw)
            if kind == "dead":
                st["dead"] = True
            elif kind == "sched":
                st["sched"] += 1
            if st["calls"] > calls_min(pid):
                st["dead"] = True  # past the smallest budget: it is spent
            log("provider %s refused attempt %d with HTTP %d" % (pid, st["calls"], status))
        if api == "anthropic":
            return 502, "application/json", json.dumps({
                "type": "error", "error": {"type": "api_error",
                "message": "no free model could serve this request"}}).encode()
        return 502, "application/json", json.dumps({
            "error": {"message": "no free model could serve this request",
                      "type": "api_error", "code": None}}).encode()


AGG = None  # set in main


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, status, ctype, payload):
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _send_json(self, status, obj):
        self._send(status, "application/json", json.dumps(obj).encode())

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/v1/models" and AGG is not None:
            anthropic_shape = self.headers.get("x-api-key") is not None
            self._send_json(200, AGG.models_body(anthropic_shape))
            return
        self._send_json(404, {"error": {"message": "no such route",
                                        "type": "not_found", "code": None}})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        api = None
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        if api is None or AGG is None:
            self._send_json(404, {"error": {"message": "no such route",
                                            "type": "not_found", "code": None}})
            return
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        if len(raw) > MAX_BODY_BYTES:
            self._send_json(413, {"error": {"message": "request too large",
                                            "type": "invalid_request_error", "code": None}})
            return
        try:
            body = json.loads(raw.decode("utf-8"))
            if not _is_record(body):
                raise ValueError("not an object")
        except ValueError:
            if api == "anthropic":
                self._send_json(400, {"type": "error", "error": {
                    "type": "invalid_request_error",
                    "message": "the request body is not one JSON object"}})
            else:
                self._send_json(400, {"error": {
                    "message": "the request body is not one JSON object",
                    "type": "invalid_request_error", "code": None}})
            return
        status, ctype, payload = AGG.serve(api, body)
        self._send(status, ctype, payload)


# ----------------------------------------------------------------- main

def load_probes_and_catalog(brief):
    probes = [probe_provider(p) for p in brief["providers"]]
    return probes, build_catalog(brief, probes)


def register(port, catalog):
    status, _ctype, raw = http_call("POST", ENV_URL + "/register", {},
                                    {"protocol_version": PROTOCOL_VERSION,
                                     "port": port, "catalog": catalog}, timeout=5.0)
    if status != 200:
        raise RuntimeError("register answered HTTP %d: %s"
                           % (status, raw[:200].decode("utf-8", "replace")))
    reply = json.loads(raw.decode("utf-8"))
    log("registered: %s requests" % reply.get("requests"))


def poll_until_done():
    while True:
        try:
            status, _ctype, raw = http_call("GET", ENV_URL + "/work", {}, None,
                                            timeout=2.0)
            if status == 200:
                work = json.loads(raw.decode("utf-8"))
                if _is_record(work) and work.get("done") is True:
                    return
        except Exception:
            pass
        time.sleep(0.1)


def main():
    global AGG
    brief = read_brief(PROMPT_FILE)
    if "--list" in sys.argv:
        if ENV_URL:
            _probes, catalog = load_probes_and_catalog(brief)
        else:
            catalog = {"providers": []}
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    probes, catalog = load_probes_and_catalog(brief)
    AGG = Aggregator(brief, probes, catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked the env's port; cannot start")
        return 1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log("listening on 0.0.0.0:%d with %d providers and %d free models"
        % (port, len(catalog["providers"]), len(AGG.served_models())))
    register(port, catalog)
    poll_until_done()
    log("workload done; exiting")
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
