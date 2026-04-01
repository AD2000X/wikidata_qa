# Wikidata QA

A Python-based question answering system that retrieves factual answers from Wikidata using SPARQL queries. The project contains four implementations arranged by complexity: a minimal single-file version, an advanced version with caching and disambiguation, an LLM-assisted version with Gemini-based semantic parsing, and a dynamic LLM version with fully dynamic entity linking.

## Overview

This system takes natural language questions (e.g. "How old is Tom Cruise?", "What is the capital of France?") and returns structured answers by querying the Wikidata knowledge graph. The core pipeline follows three main steps: intent detection, entity resolution, and SPARQL execution.

Each implementation builds on the previous one:

- **wikidata_qa_min** -- Minimal prototype (~90 lines). Regex intent matching, hardcoded QID lookups, two intent types only. Question -> rule-based intent recognition -> entity-to-QID mapping -> SPARQL query construction -> Wikidata query -> formatted answer.

- **wikidata_qa_adv** -- Advanced version (~920 lines). Multi-signal entity disambiguation, persistent caching, retry with exponential backoff, async batch queries, and multi-hop SPARQL reasoning. Question -> intent detection -> entity extraction -> candidate search and disambiguation -> QID selection -> SPARQL query construction -> cached/retried Wikidata query execution -> result formatting.

- **wikidata_qa_llm** -- LLM-assisted version. Adds Gemini 2.5 Flash-Lite semantic parsing on top of the deterministic pipeline, while keeping the legacy regex parser as a fallback. All downstream modules (entity resolution, SPARQL generation, execution verification, and answer formatting) remain deterministic. Question -> Gemini semantic parsing -> frame validation -> confidence-based routing / regex fallback -> entity resolution and disambiguation -> type reconciliation -> SPARQL query construction -> cached/retried Wikidata query execution -> execution verification -> result formatting.

- **wikidata_qa_llm_dyna** -- Dynamic LLM-assisted version. Keeps the semantic parsing, audit trail, and deterministic SPARQL compilation of the LLM version, but replaces fixed or semi-manual entity resolution with a fully dynamic entity linking pipeline. Candidate recall is generated at runtime, ranking combines embedding similarity with context-aware disambiguation, and low-confidence candidates are rejected through NIL detection. Question -> Gemini semantic parsing / regex fallback -> frame validation -> dynamic candidate recall -> embedding-based entity linking -> type reconciliation -> deterministic SPARQL query construction -> cached/retried Wikidata query execution -> execution verification -> result formatting.

## Quick Start

### Requirements

- Python 3.10+
- SPARQLWrapper
- httpx (advanced and LLM versions)
- diskcache (advanced and LLM versions)

```bash
pip install sparqlwrapper httpx diskcache
```

### Gemini API Key (LLM versions only)

The `wikidata_qa_llm` and `wikidata_qa_llm_dyna` versions require a free Gemini API key from [Google AI Studio](https://aistudio.google.com/). Without the key, the system falls back to the regex parser automatically. In the dynamic version, the same key also enables Gemini embedding-based entity ranking.

**Linux / macOS:**

```bash
export GEMINI_API_KEY="your-key-here"
```

To make it persistent, add the line above to `~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`, then run `source ~/.bashrc` (or the relevant file).

**Windows (PowerShell):**

```powershell
$env:GEMINI_API_KEY="your-key-here"
```

To make it persistent, use:

```powershell
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-key-here", "User")
```

**Windows (Command Prompt):**

```cmd
set GEMINI_API_KEY=your-key-here
```

To make it persistent, use:

```cmd
setx GEMINI_API_KEY "your-key-here"
```

After using `setx`, restart the terminal for the change to take effect.

### Running Tests

Run them from the project root directory:

```bash
# Minimal version (direct run / requires network access to Wikidata)
cd wikidata_qa_min
python wikidata_qa_min.py

# Advanced version (requires network access to Wikidata)
cd ../wikidata_qa_adv
python test_wikidata_qa_adv.py

# LLM version - end-to-end assertions (network access required;
# uses Gemini if GEMINI_API_KEY is set, otherwise falls back to regex)
cd ../wikidata_qa_llm
python test_wikidata_qa_llm.py

# LLM version - semantic pipeline tests
# unit-style validator/reconciler/verifier tests run without a key;
# Gemini integration tests are skipped unless GEMINI_API_KEY is set
python test_semantic_pipeline.py

# Dynamic LLM version - four-question regression test
cd ../wikidata_qa_llm_dyna
python test_wikidata_qa_llm.py

# Dynamic LLM version - semantic pipeline tests
python test_semantic_pipeline.py

# LLM version - force Gemini-enabled end-to-end run
GEMINI_API_KEY="your-key" python test_wikidata_qa_llm.py
```

On Windows PowerShell, set the environment variable before running:

```powershell
$env:GEMINI_API_KEY="your-key-here"
python test_wikidata_qa_llm.py
```

For the dynamic LLM version on Windows PowerShell:

```powershell
cd .\wikidata_qa_llm_dyna
$env:GEMINI_API_KEY="your-key-here"
python test_wikidata_qa_llm.py
python test_semantic_pipeline.py
```

## Project Structure

```text
SPARQL/
├── requirements.txt
├── wikidata_qa_min/
│   └── wikidata_qa_min.py
├── wikidata_qa_adv/
│   ├── config.py
│   ├── qid_fallback.json
│   ├── test_wikidata_qa_adv.py
│   └── wikidata_qa_adv.py
├── wikidata_qa_llm/
│   ├── config.py
│   ├── qid_fallback.json
│   ├── gemini_config.py
│   ├── semantic_parser.py
│   ├── resolution_reconciler.py
│   ├── execution_verifier.py
│   ├── wikidata_qa_llm.py
│   ├── test_wikidata_qa_llm.py
│   └── test_semantic_pipeline.py
└── wikidata_qa_llm_dyna/
    ├── config.py
    ├── gemini_config.py
    ├── semantic_parser.py
    ├── entity_linker.py
    ├── resolution_reconciler.py
    ├── execution_verifier.py
    ├── wikidata_qa_llm.py
    ├── test_wikidata_qa_llm.py
    └── test_semantic_pipeline.py
```

## Supported Question Types

| Intent | Example |
|---|---|
| Age | How old is Madonna? |
| Population | What is the population of New York? |
| Capital | What is the capital of France? |
| Spouse birth place | Where was the spouse of Tom Hanks born? |
| Birth country capital | What is the capital of the country where Einstein was born? |
| Spouse occupation | What is the occupation of the spouse of Barack Obama? |
| Cities in country | List cities in Japan |
| Actors born in place | Which actors were born in London? |
| Humans with occupation | List people who work as physicists |

Note: The minimal version only supports age and population queries.

## Usage

### Minimal Version

```python
from wikidata_qa_min import ask

answer = ask("How old is Tom Cruise?")
print(answer)  # e.g. "62"
```

### Advanced Version

```python
from wikidata_qa_adv import WikidataQA

with WikidataQA() as qa:
    result = qa.ask("What is the capital of France?")
    print(result["answer"])       # "Paris"
    print(result["intent"])       # "capital"
    print(result["resolved_qid"]) # "Q142"
```

The advanced version returns a dictionary containing the detected intent, resolved entity, generated SPARQL query, and raw result rows, which is useful for debugging and auditing.

### LLM Version

```python
from wikidata_qa_llm import WikidataQA

with WikidataQA() as qa:
    result = qa.ask("What is the capital of France?")
    print(result["answer"])        # "Paris"
    print(result["parser_source"]) # "gemini" or "regex_fallback"
    print(result["status"])        # "ok"
```

To use regex-only mode without any LLM dependency:

```python
from wikidata_qa_llm import WikidataQA

with WikidataQA(use_llm_parser=False) as qa:
    result = qa.ask("How old is Tom Cruise?")
    print(result["answer"])
```

### Dynamic LLM Version

Inside `wikidata_qa_llm_dyna/`, the dynamic version exposes the same high-level API:

```python
from wikidata_qa_llm import WikidataQA

with WikidataQA() as qa:
    result = qa.ask("What is the population of New York?")
    print(result["answer"])                   # e.g. "8804190"
    print(result["resolved_qid"])             # e.g. "Q60"
    print(result["parser_source"])            # "gemini" or "regex_fallback"
    print(result["linking"]["score"])         # entity-linking score
    print(result["linking"]["candidate_scores"])
```

To use regex-only parsing mode in the dynamic version:

```python
from wikidata_qa_llm import ask

answer = ask("How old is Tom Cruise?", use_llm_parser=False)
print(answer)
```

To run the basic four-question regression test in the dynamic version:

```bash
cd wikidata_qa_llm_dyna
python test_wikidata_qa_llm.py
```

### Async Batch Queries

Both the LLM and dynamic LLM versions support asynchronous batch processing:

```python
import asyncio
from wikidata_qa_llm import ask_batch

questions = [
    "How old is Madonna?",
    "What is the population of London?",
    "What is the capital of Germany?",
]

results = asyncio.run(ask_batch(questions))
for r in results:
    print(f"{r['question']} -> {r['answer']} (via {r['parser_source']})")
```

### Inspecting the Audit Trail (LLM Versions)

```python
from wikidata_qa_llm import WikidataQA

with WikidataQA() as qa:
    result = qa.ask("Where was Tom Hanks's spouse born?")
    print(result["semantic_frame"])       # {"intent": "spouse_birth_place", ...}
    print(result["parser_source"])        # "gemini"
    print(result["validation_warnings"])  # []
    print(result["reconciliation"])       # {"is_compatible": True, ...}
    print(result["verification"])         # {"is_valid": True, ...}
    print(result["linking"])              # entity-linking diagnostics
```

## Architecture

### Shared Pipeline

All four versions follow the same high-level structure:

- detect an intent from the question
- resolve the entity to a Wikidata item
- compile a deterministic SPARQL query from intent + QID
- execute the query and format the final answer

The main differences are in how intent detection and entity resolution are implemented, and how much runtime metadata each version preserves.

### Intent Detection

Shared concept:
- every version uses a controlled intent inventory rather than open-ended query generation

Version differences:
- `wikidata_qa_min` uses regex pattern matching only
- `wikidata_qa_adv` uses rule-based keyword matching first, then a lightweight semantic fallback
- `wikidata_qa_llm` and `wikidata_qa_llm_dyna` add a Gemini-based semantic parser that outputs a structured `SemanticFrame` (`intent`, `entity_text`, `entity_type_hint`, `confidence`) under a constrained schema
- `wikidata_qa_llm` and `wikidata_qa_llm_dyna` retain the regex parser as a deterministic fallback, with confidence-based routing controlling when Gemini is trusted

Confidence routing in the LLM versions:

- `>= 0.75`: use Gemini directly
- `0.40` to `0.75`: try regex first, then fall back to Gemini
- `< 0.40`: prefer regex; if regex also fails, accept Gemini but mark the result as ambiguous
- `< 0.20` with `intent=unsupported`: reject the question

### Entity Resolution

Shared concept:
- every version maps the extracted entity text to a Wikidata QID before SPARQL is generated

Version differences:
- `wikidata_qa_min` uses a small hardcoded dictionary plus exact label lookup
- `wikidata_qa_adv` and `wikidata_qa_llm` query the Wikidata Search API, retrieve candidates, and rank them with deterministic signals such as label overlap, alias matching, type compatibility, description relevance, and disambiguation penalties
- `wikidata_qa_llm_dyna` replaces that resolver with a dynamic entity linking pipeline: candidate recall from Wikidata Search API plus alias expansion, embedding-based ranking with Gemini embeddings, context-aware disambiguation using the full question text, and NIL detection when the top score falls below a threshold

### SPARQL Compilation And Answering

Shared concept:
- all four versions use deterministic template-based SPARQL generation from intent and QID
- the model never generates SPARQL directly

Version differences:
- `wikidata_qa_min` supports only a small subset of intents and uses a minimal answer layer
- `wikidata_qa_adv`, `wikidata_qa_llm`, and `wikidata_qa_llm_dyna` support a larger intent set, including multi-hop templates and richer answer formatting

### Validation, Reconciliation, And Verification

Shared concept:
- later versions add explicit guardrails between parsing and answer generation rather than trusting intermediate outputs blindly

Version differences:
- `wikidata_qa_min` and `wikidata_qa_adv` do not expose a separate semantic-frame validation layer
- `wikidata_qa_llm` and `wikidata_qa_llm_dyna` use `FrameValidator` to catch hard parser errors and record soft warnings
- `wikidata_qa_llm` and `wikidata_qa_llm_dyna` use `ResolutionReconciler` to compare parser type hints with resolved entity types
- `wikidata_qa_llm` and `wikidata_qa_llm_dyna` use `ExecutionVerifier` to validate SPARQL results before answer formatting

### Caching, Retry, And Concurrency

Shared concept:
- all versions query public Wikidata infrastructure and therefore depend on external service availability

Version differences:
- `wikidata_qa_min` keeps the runtime simple and does not include a persistent cache layer
- `wikidata_qa_adv`, `wikidata_qa_llm`, and `wikidata_qa_llm_dyna` use `diskcache` for persistent caching, exponential backoff with jitter for retryable HTTP failures, and asyncio semaphores for bounded concurrency

### Audit Trail

Shared concept:
- all versions ultimately return a factual answer string

Version differences:
- `wikidata_qa_min` exposes only the final answer
- `wikidata_qa_adv` adds resolved entity information, generated SPARQL, and raw rows
- `wikidata_qa_llm` and `wikidata_qa_llm_dyna` add full audit information such as semantic frame, parser source, validation warnings, reconciliation output, verification output, and linking diagnostics

## Design Principles

### Shared Across All Versions

- **Deterministic query generation.** SPARQL is built from known intent templates rather than being generated freely from the model.
- **Composable pipeline design.** Intent detection, entity resolution, query construction, and answer formatting remain separable stages.
- **Pragmatic scope control.** The system answers a constrained class of factual questions rather than attempting unrestricted dialogue over Wikidata.

### Shared By `wikidata_qa_adv`, `wikidata_qa_llm`, And `wikidata_qa_llm_dyna`

- **Operational robustness.** Caching, retries, and bounded concurrency are treated as part of the core system rather than optional wrappers.
- **Entity disambiguation as a first-class problem.** Later versions spend more logic on picking the right QID before query execution.

### Shared By `wikidata_qa_llm` And `wikidata_qa_llm_dyna`

- **LLM chooses among allowed structures; it never defines the structure.** Gemini produces a constrained semantic frame, while entity resolution, SPARQL generation, and answer formatting stay deterministic.
- **Graceful degradation.** If Gemini is unavailable or low-confidence, the system falls back to the legacy regex parsing logic while preserving the audit structure.
- **Auditability.** Successful supported responses include a full decision trace, including which parser produced the result and why; unsupported responses still retain parsing-stage audit fields.
- **Provider-agnostic semantic parsing.** The `SemanticParser` protocol can be reimplemented for other model providers without changing the rest of the pipeline.

### Specific To `wikidata_qa_llm_dyna`

- **Dynamic linking without hardcoded QIDs.** The dynamic version resolves entities through runtime recall, ranking, and NIL detection instead of fixed per-entity overrides.
- **Context-sensitive disambiguation.** Entity ranking should depend on the full question, not just label overlap.

## Limitations

### Shared Across All Versions

- The system queries the public Wikidata SPARQL endpoint, which has rate limits and may occasionally return timeout errors.
- Multi-hop reasoning is template-based. Each property chain (e.g. person -> spouse -> birth place) is hardcoded in the SPARQL builder, so adding new intents requires code changes.

### Shared By `wikidata_qa_adv`, `wikidata_qa_llm`, And `wikidata_qa_llm_dyna`

- Entity resolution quality still depends on candidate recall from external Wikidata services. If the right candidate is not recalled, ranking cannot recover it.

### Shared By `wikidata_qa_llm` And `wikidata_qa_llm_dyna`

- Intent detection still depends on predefined patterns (regex) or LLM classification (Gemini), so complex or heavily paraphrased questions may fail at the parsing stage.
- Gemini usage limits and API availability depend on the current Google AI Studio / Gemini API policy.

### Specific To `wikidata_qa_min`

- The minimal version relies on hardcoded entity coverage and a very small supported intent set.
- Hardcoded assertions in the minimal version will become outdated as real-world data changes over time.

### Specific To `wikidata_qa_llm_dyna`

- The dynamic version depends more heavily on external APIs because semantic parsing, embedding-based ranking, Wikidata candidate recall, and Wikidata lookup all contribute to the final answer path.
- Dynamic entity linking is more flexible than fixed QID maps, but it also introduces more moving parts and more ways for ambiguous entities to fail or be rejected as NIL.
