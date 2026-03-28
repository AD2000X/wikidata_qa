from wikidata_qa_adv import ask as core_ask


def test_basic_assertions():
    assert "63" == ask("how old is Tom Cruise")
    assert "67" == ask("what age is Madonna?")
    assert "8799728" == ask("what is the population of London")
    assert "8804190" == ask("what is the population of New York?")


def ask(question: str, endpoint: str = 'https://query.wikidata.org/sparql'):
    return core_ask(question, endpoint=endpoint)


if __name__ == '__main__':
    assert '63' == ask('how old is Tom Cruise')
    assert '67' == ask('what age is Madonna?')
    assert '8799728' == ask('what is the population of London')
    assert '8804190' == ask('what is the population of New York?')
    print('All assertions passed')
