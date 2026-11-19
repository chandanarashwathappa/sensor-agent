from agent.analytics import DATA_PATH
from evaluation.checks import check_keywords, check_number, numbers_in
from evaluation.questions import build_questions


def test_numbers_in_handles_separators_and_decimals():
    assert numbers_in("10,000 records, 3.39 % failed, -2 offset") == [10000.0, 3.39, -2.0]


def test_check_number_tolerance():
    assert check_number("The average is 46.08 Nm", 46.08, 0.1)
    assert check_number("about 46.1 Nm", 46.08, 0.1)
    assert not check_number("about 47 Nm", 46.08, 0.1)
    assert check_number("339 machines failed", 339, 0)
    assert not check_number("340 machines failed", 339, 0)


def test_check_keywords_single_letter_is_a_whole_word():
    assert check_keywords("Type L has the highest rate", [["L", "low"]])
    assert check_keywords("low-quality products", [["L", "low"]])
    assert not check_keywords("Medium (M) products", [["L", "low"]])     # 'L' inside a word doesn't count


def test_question_set():
    qs = build_questions(DATA_PATH)
    assert len(qs) == 20
    assert qs[1]["expected"] == 339
    assert sum(q["kind"] == "number" for q in qs) == 16
