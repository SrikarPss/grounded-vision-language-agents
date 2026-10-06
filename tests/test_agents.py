from pathlib import Path

from grounded_vla.agents import ORAAgent, ReActAgent, SingleShotVLMAgent
from grounded_vla.backends.mock import MockBackend
from grounded_vla.env import StaticQAEnv, TaskReplayEnv
from grounded_vla.schemas import Action, ActionType, Observation, Task


FIXT = Path(__file__).parent.parent / "data" / "samples" / "images"


def _click_task() -> Task:
    return Task(
        task_id="t1",
        instruction='Click the login button. [[GT_ACTION: {"type": "click", "target": "#login-button"}]]',
        source="synthetic",
        initial_observation=Observation(step=0, image_path=FIXT / "login.png"),
        gold_actions=[Action(type=ActionType.CLICK, target="#login-button")],
        max_steps=3,
    )


def _answer_task() -> Task:
    return Task(
        task_id="t2",
        instruction='What molecule is shown? [[GT_ACTION: {"type": "answer", "value": "water"}]]',
        source="scienceqa",
        initial_observation=Observation(step=0, image_path=FIXT / "diagram.png"),
        gold_actions=[Action(type=ActionType.ANSWER, value="water")],
        gold_answer="water",
        max_steps=2,
    )


def test_ora_agent_reaches_terminal_action_with_oracle_mock():
    agent = ORAAgent(MockBackend(policy="oracle", supports_vision=True))
    task = _answer_task()
    env = StaticQAEnv()
    traj = agent.run(task, env)
    assert traj.terminated
    assert traj.final_answer == "water"


def test_ora_agent_requires_vision_backend():
    from grounded_vla.backends.mock import MockBackend as MB

    text_only = MB(supports_vision=False)
    try:
        ORAAgent(text_only)
    except ValueError:
        return
    raise AssertionError("ORAAgent should reject text-only backends")


def test_react_agent_runs_without_images():
    agent = ReActAgent(MockBackend(policy="oracle", supports_vision=False))
    task = _answer_task()
    env = StaticQAEnv()
    traj = agent.run(task, env)
    assert traj.terminated
    assert traj.final_answer == "water"


def test_single_shot_agent_emits_one_step():
    agent = SingleShotVLMAgent(MockBackend(policy="oracle", supports_vision=True))
    task = _click_task()
    env = TaskReplayEnv()
    traj = agent.run(task, env)
    assert len(traj.steps) == 1


def test_ora_visual_re_encoding_changes_outputs_between_frames(tmp_path):
    """Smoke check for H2's mechanism: different frames -> different rationales."""
    # Build two tasks with distinct images; mock backend embeds a fingerprint
    # of the image into its 'Thought' line, so different frames should
    # yield different rationales.
    agent = ORAAgent(MockBackend(policy="oracle", supports_vision=True))
    t1 = _click_task()
    t2 = Task(
        task_id="t3",
        instruction='Press Submit. [[GT_ACTION: {"type": "click", "target": "#submit"}]]',
        source="synthetic",
        initial_observation=Observation(step=0, image_path=FIXT / "submit.png"),
        gold_actions=[Action(type=ActionType.CLICK, target="#submit")],
    )
    tr1 = agent.run(t1, TaskReplayEnv())
    tr2 = agent.run(t2, TaskReplayEnv())
    r1 = tr1.steps[0].action.rationale or ""
    r2 = tr2.steps[0].action.rationale or ""
    assert r1 != r2, f"image fingerprint did not vary: {r1!r} vs {r2!r}"
