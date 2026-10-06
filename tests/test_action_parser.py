from grounded_vla.action_parser import parse
from grounded_vla.schemas import ActionType


def test_parses_clean_json_action():
    text = (
        "Thought: I'll click the login button.\n"
        'Action: {"type": "click", "target": "#login-button"}\n'
    )
    r = parse(text)
    assert r.ok
    assert r.action.type == ActionType.CLICK
    assert r.action.target == "#login-button"
    assert "login" in r.rationale.lower()


def test_parses_fenced_json():
    text = (
        "Thought: let me type the password.\n"
        "Action: ```json\n"
        '{"type": "type", "target": "#pw", "value": "hunter2"}\n'
        "```\n"
    )
    r = parse(text)
    assert r.ok
    assert r.action.type == ActionType.TYPE
    assert r.action.target == "#pw"
    assert r.action.value == "hunter2"


def test_falls_back_to_nl_on_bad_json():
    text = (
        "Thought: I should click the button.\n"
        "Action: click on the login button.\n"
    )
    r = parse(text)
    assert r.ok
    assert r.action.type == ActionType.CLICK
    assert "login" in r.action.target.lower()


def test_parses_answer_action():
    text = "Thought: the diagram shows water.\nAction: Answer: water\n"
    r = parse(text)
    assert r.ok
    assert r.action.type == ActionType.ANSWER
    assert r.action.value.strip().lower() == "water"


def test_empty_response_is_error():
    r = parse("")
    assert not r.ok
    assert r.action is None
    assert r.error


def test_unparseable_response_is_error():
    r = parse("I am thinking... but not emitting anything structured.")
    assert not r.ok
    assert r.action is None
