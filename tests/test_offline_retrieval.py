from benchmarks.offline_retrieval import parse_squad, split_words, recall_at_k, optimize
import pytest

# Schema fixture validates parser/retriever behaviour only.
# Production metrics ALWAYS download official SQuAD questions and contexts.
DOCUMENTS = [
    ("Which planet has rings?", "Saturn has prominent rings of ice."),
    ("What has a rocky surface?", "Mars is a rocky world with craters."),
    ("Which planet is closest to the Sun?", "Mercury is closest to the Sun."),
    ("What color is Neptune?", "Neptune has a blue atmosphere."),
    ("Which is the largest planet?", "Jupiter is the largest planet."),
    ("Which planet do humans live on?", "Earth supports human life."),
    ("Which is the hottest planet?", "Venus has a dense hot atmosphere."),
    ("Which is a dwarf planet?", "Pluto is classified as a dwarf planet."),
    ("What rotates on its side?", "Uranus has an extreme axial tilt."),
    ("Which planet has two small moons?", "Mars has Phobos and Deimos."),
]


def test_parser_accepts_squad_structure():
    document = {"data": [{"paragraphs": [{"context": "Published paragraph", "qas": [{"question": "Which paragraph?", "answers": [{"text": "Published", "answer_start": 0}]}]}]}]}
    assert parse_squad(document) == [("Which paragraph?", "Published paragraph")]


def test_chunks():
    assert split_words("one two three", 2) == ["one two", "three"]


def test_retrieval_and_fixed_seed():
    contexts = [context for question, context in DOCUMENTS]
    assert 0 <= recall_at_k(DOCUMENTS, contexts, 8, 2) <= 1
    a = optimize(DOCUMENTS, generations=2, population_size=3, seed=42)
    b = optimize(DOCUMENTS, generations=2, population_size=3, seed=42)
    assert a == b
    assert a["questions_train"] + a["questions_heldout"] == 10


def test_no_empty_or_synthetic_fallback():
    with pytest.raises(ValueError):
        parse_squad({"data": []})
    with pytest.raises(ValueError):
        optimize([])
