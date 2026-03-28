from wikidata_qa_llm import WikidataQA


def core_ask(question: str, endpoint: str = "https://query.wikidata.org/sparql") -> str:
    with WikidataQA() as qa:
        return qa.ask(question, endpoint=endpoint)["answer"]


def test_basic_assertions():
    assert "63" == ask("how old is Tom Cruise")
    assert "67" == ask("what age is Madonna?")
    assert "8799728" == ask("what is the population of London")
    assert "8804190" == ask("what is the population of New York?")


def ask(question: str, endpoint: str = "https://query.wikidata.org/sparql"):
    return core_ask(question, endpoint=endpoint)


if __name__ == "__main__":
    assert "63" == ask("how old is Tom Cruise")
    assert "67" == ask("what age is Madonna?")
    assert "8799728" == ask("what is the population of London")
    assert "8804190" == ask("what is the population of New York?")
    print("All assertions passed")