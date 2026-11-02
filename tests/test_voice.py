from whiteout.voice import parse_dictation, words_to_digits


def test_words_to_digits():
    assert words_to_digits("point two five one two") == ".2512"
    assert "2.501" in words_to_digits("two point five zero one")
    assert words_to_digits("fourteen") == "14"
    assert words_to_digits("twenty three") == "23"


def test_parse_real_transcript():
    t = "balloon fourteen, point two five one two balloon three, two point five zero one."
    assert parse_dictation(t, ["14", "3", "9.2"]) == [("14", ".2512"), ("3", "2.501")]


def test_rejects_unknown_rows_and_handles_ok():
    assert parse_dictation("balloon ninety nine point five", ["14"]) == []
    assert parse_dictation("balloon 7 conforms", ["7"]) == [("7", "OK")]
    assert parse_dictation("balloon 9.2 0.25", ["9.2"]) == [("9.2", "0.25")]
