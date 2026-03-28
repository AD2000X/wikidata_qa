# Wikidata QA

A Python-based question answering system that retrieves factual answers from Wikidata using SPARQL queries. The project contains three implementations arranged by complexity: a minimal single-file version, an advanced version with caching and disambiguation, and an LLM-assisted version with Gemini-based semantic parsing.

## Overview

This system takes natural language questions (e.g. "How old is Tom Cruise?", "What is the capital of France?") and returns structured answers by querying the Wikidata knowledge graph. The core pipeline follows three main steps: intent detection, entity resolution, and SPARQL execution.

Each implementation builds on the previous one:

- **wikidata_qa_min** -- Minimal prototype (~90 lines). Regex intent matching, hardcoded QID lookups, two intent types only. Question → rule-based intent recognition → entity-to-QID mapping → SPARQL query construction → Wikidata query → formatted answer.

- **wikidata_qa_adv** -- Advanced version (~920 lines). Multi-signal entity disambiguation, persistent caching, retry with exponential backoff, async batch queries, and multi-hop SPARQL reasoning. Question → intent detection → entity extraction → candidate search and disambiguation → QID selection → SPARQL query construction → cached/retried Wikidata query execution → result formatting.

- **wikidata_qa_llm** -- LLM-assisted version. Replaces the regex-based intent detection with Gemini 2.5 Flash-Lite structured output, while keeping the existing regex pipeline as a deterministic fallback. All downstream modules (entity resolution, SPARQL generation, answer formatting) remain fully deterministic. Question → intent detection (rule-based + semantic fallback) → entity extraction → candidate search and disambiguation → QID selection → SPARQL query construction (including multi-hop property chains) → cached/retried Wikidata query execution → result formatting.

## Quick Start

### Requirements

- Python 3.10+
- SPARQLWrapper
- httpx (advanced and LLM versions)
- diskcache (advanced and LLM versions)

```bash
pip install sparqlwrapper httpx diskcache
```

### Gemini API Key (LLM version only)

The LLM version requires a free Gemini API key from [Google AI Studio](https://aistudio.google.com/). Without the key, the system falls back to the regex parser automatically.

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

# LLM version — unit tests (no API key needed)
cd ../wikidata_qa_llm
python test_wikidata_qa_llm.py

# LLM version — semantic pipeline tests (requires GEMINI_API_KEY)
python test_semantic_pipeline.py

# LLM version — full integration tests (requires GEMINI_API_KEY + network)
GEMINI_API_KEY="your-key" python test_wikidata_qa_llm.py
```

On Windows PowerShell, set the environment variable before running:

```powershell
$env:GEMINI_API_KEY="your-key-here"
python test_wikidata_qa_llm.py
```

## Project Structure

```
SPARQL/
├── requirements.txt
├── wikidata_qa_min/
│   └── wikidata_qa_min.py
├── wikidata_qa_adv/
│   ├── config.py
│   ├── qid_fallback.json
│   ├── test_wikidata_qa_adv.py
│   └── wikidata_qa_adv.py
└── wikidata_qa_llm/
    ├── config.py
    ├── qid_fallback.json
    ├── gemini_config.py
    ├── semantic_parser.py
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
    print(result["answer"])          # "Paris"
    print(result["parser_source"])   # "gemini" or "regex_fallback"
    print(result["status"])          # "ok"
```

To use regex-only mode without any LLM dependency:

```python
from wikidata_qa_llm import WikidataQA

with WikidataQA(use_llm_parser=False) as qa:
    result = qa.ask("How old is Tom Cruise?")
    print(result["answer"])
```

### Async Batch Queries

Both the advanced and LLM versions support asynchronous batch processing:

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

### Inspecting the Audit Trail (LLM Version)

```python
from wikidata_qa_llm import WikidataQA

with WikidataQA() as qa:
    result = qa.ask("Where was Tom Hanks's spouse born?")
    print(result["semantic_frame"])       # {"intent": "spouse_birth_place", ...}
    print(result["parser_source"])        # "gemini"
    print(result["validation_warnings"])  # []
    print(result["reconciliation"])       # {"is_compatible": True, ...}
    print(result["verification"])         # {"is_valid": True, ...}
```

## Architecture

### Intent Detection

The minimal version uses a list of regex patterns to match questions to predefined intents. The advanced version adds a two-stage approach: rule-based keyword matching first, then token-level semantic similarity as a fallback if no rule matches.

The LLM version replaces this with a Gemini-based semantic parser that outputs a structured SemanticFrame (intent, entity_text, entity_type_hint, confidence) using constrained JSON schema at the API level. The existing regex pipeline serves as a deterministic fallback.

Confidence routing in the LLM version:

- >= 0.75: use Gemini result directly
- 0.40 to 0.75: try regex first, fall back to Gemini
- < 0.40: prefer regex; if regex also fails, accept Gemini but mark as ambiguous
- < 0.20 with intent=unsupported: refuse the question

### Entity Resolution

The minimal version relies on a hardcoded dictionary and exact label matching via SPARQL. The advanced and LLM versions query the Wikidata Search API, retrieve up to eight candidates, and score each one based on label similarity, alias matching, type hierarchy checking (P31/P279), description relevance, and disambiguation penalties. The highest-scoring candidate is selected.

### Frame Validation (LLM Version)

FrameValidator checks the SemanticFrame for hard errors (invalid intent, empty entity, clearly contradictory type hints) and soft warnings (unknown type hint, low/medium confidence). The entity_type_hint is treated as a weak signal from the parser -- only obviously contradictory combinations trigger hard errors.

### Resolution Reconciliation (LLM Version)

After entity resolution, the ResolutionReconciler compares the parser's type hint against the resolver's actual types and the intent's expected types. It corrects the type hint for the audit trail and logs warnings, but never hard-rejects a result.

### SPARQL Compilation

All three versions use deterministic template-based SPARQL generation from intent and QID. In the LLM version, the model never generates SPARQL directly.

### Execution Verification (LLM Version)

ExecutionVerifier checks SPARQL results for empty results, missing fields, invalid values, and cardinality violations before answer formatting.

### Caching and Retry

The advanced and LLM versions use diskcache for persistent caching with configurable TTL per intent. HTTP requests include exponential backoff with jitter and a retryable status code whitelist (408, 429, 500, 502, 503, 504). Concurrency is controlled through asyncio semaphores.

### Audit Trail (LLM Version)

Every response includes a full decision trace: semantic frame, parser source, validation errors and warnings, reconciliation result, and verification result.

## Design Principles

- **LLM chooses among allowed structures; it never defines the structure.** The model outputs a constrained semantic frame. All formal operations (entity resolution, SPARQL generation, answer formatting) are deterministic.
- **Graceful degradation.** If Gemini is unavailable, the system falls back to the existing regex pipeline with identical behaviour to the pre-LLM version.
- **Auditability.** Every response includes the full decision trace, including which parser produced the result and why.
- **Provider-agnostic.** The SemanticParser protocol can be implemented for any LLM provider (Gemini, OpenRouter, local models) without changing the pipeline.

## Limitations

- Intent detection depends on predefined patterns (regex) or LLM classification (Gemini). Complex or heavily paraphrased questions may not be handled.
- Entity resolution uses token-level overlap rather than embedding-based similarity, so it may struggle with ambiguous short names.
- The system queries the public Wikidata SPARQL endpoint, which has rate limits and may occasionally return timeout errors.
- Hardcoded assertions in the minimal version will become outdated as real-world data changes over time.
- Gemini free tier allows 15 RPM and 1,000 RPD for Flash-Lite.
- Multi-hop reasoning is template-based. Each property chain (e.g. person → spouse → birth place) is hardcoded in the SPARQL builder. Adding new multi-hop intents requires manual changes to config.py, build_sparql(), and format_answer(). A deterministic chain compiler could generalise pure-chain intents without resorting to graph matching or LLM-generated SPARQL.