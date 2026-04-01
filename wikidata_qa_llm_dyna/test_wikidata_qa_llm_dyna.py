from wikidata_qa_llm import ask


CASES = [
    ("how old is Tom Cruise", "63"),
    ("what age is Madonna?", "67"),
    ("what is the population of London", "8799728"),
    ("what is the population of New York?", "8804190"),
]


def test_basic_assertions() -> None:
    for question, expected in CASES:
        actual = ask(question, use_llm_parser=False)
        assert expected == actual, f"Question '{question}' expected '{expected}', got '{actual}'"
        print(f"[PASS] {question} -> {actual}")


if __name__ == "__main__":
    test_basic_assertions()
    print("All assertions passed")
