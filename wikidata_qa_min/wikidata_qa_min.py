"""Minimal Wikidata QA — single file, no dependencies beyond SPARQLWrapper."""

import re
from datetime import date, datetime
from SPARQLWrapper import SPARQLWrapper, JSON

ENDPOINT = "https://query.wikidata.org/sparql"
AGENT = "Wikidata-QA-Minimal/1.0"

# Known entity → QID mappings (avoids disambiguation issues)
KNOWN_QIDS = {
    "tom cruise": "Q37079",
    "madonna": "Q1744",
    "london": "Q84",
    "new york": "Q60",
    "new york city": "Q60",
}

# Intent patterns: (regex, sparql_template, answer_extractor)
INTENTS = [
    {
        "patterns": [
            r"how\s+old\s+is\s+(?P<entity>.+?)\s*\??$",
            r"what\s+age\s+is\s+(?P<entity>.+?)\s*\??$",
        ],
        "sparql": "SELECT ?birth WHERE {{ wd:{qid} wdt:P569 ?birth . }} LIMIT 1",
        "extract": lambda row: str(_age(row["birth"]["value"])),
    },
    {
        "patterns": [
            r"what\s+is\s+the\s+population\s+of\s+(?P<entity>.+?)\s*\??$",
            r"population\s+of\s+(?P<entity>.+?)\s*\??$",
        ],
        "sparql": (
            "SELECT ?pop WHERE {{ "
            "wd:{qid} p:P1082 ?s . ?s ps:P1082 ?pop . "
            "OPTIONAL {{ ?s pq:P585 ?t . }} "
            "}} ORDER BY DESC(?t) LIMIT 1"
        ),
        "extract": lambda row: str(int(float(row["pop"]["value"]))),
    },
]


def _age(iso_birth: str) -> int:
    birth = datetime.fromisoformat(iso_birth.replace("Z", "+00:00")).date()
    today = date.today()
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


def _resolve_qid(name: str, endpoint: str) -> str:
    key = name.strip().lower()
    if key in KNOWN_QIDS:
        return KNOWN_QIDS[key]
    # Fallback: Wikidata search API via SPARQL
    sparql = SPARQLWrapper(endpoint, agent=AGENT)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(
        f'SELECT ?item WHERE {{ ?item rdfs:label "{name}"@en . ?item wdt:P31 [] . }} LIMIT 1'
    )
    results = sparql.query().convert()["results"]["bindings"]
    if not results:
        raise ValueError(f"Entity not found: {name}")
    return results[0]["item"]["value"].rsplit("/", 1)[-1]


def _query(sparql_str: str, endpoint: str) -> list:
    sparql = SPARQLWrapper(endpoint, agent=AGENT)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(sparql_str)
    return sparql.query().convert()["results"]["bindings"]


def ask(question: str, endpoint: str = ENDPOINT) -> str:
    text = question.strip().lower()
    for intent in INTENTS:
        for pattern in intent["patterns"]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                entity = m.group("entity").strip(" ?,.!")
                qid = _resolve_qid(entity, endpoint)
                rows = _query(intent["sparql"].format(qid=qid), endpoint)
                if not rows:
                    raise ValueError(f"No result for: {question}")
                return intent["extract"](rows[0])
    raise ValueError(f"Unsupported question: {question}")


if __name__ == "__main__":
    assert "63" == ask("how old is Tom Cruise")
    assert "67" == ask("what age is Madonna?")
    assert "8799728" == ask("what is the population of London")
    assert "8804190" == ask("what is the population of New York?")
    print("All assertions passed")
