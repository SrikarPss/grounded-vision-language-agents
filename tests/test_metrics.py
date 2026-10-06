from grounded_vla.eval.metrics import (
    mind2web_task_success,
    scienceqa_task_success,
    synthetic_task_success,
)
from grounded_vla.schemas import Action, ActionType, Observation, Task, Trajectory, TrajectoryStep


def _obs():
    return Observation(step=0)


def test_scienceqa_normalization():
    task = Task(
        task_id="q1",
        instruction="...",
        source="scienceqa",
        initial_observation=_obs(),
        gold_actions=[Action(type=ActionType.ANSWER, value="a plant cell")],
        gold_answer="A plant cell.",
    )
    traj = Trajectory(task_id="q1", final_answer="the plant cell")
    assert scienceqa_task_success(task, traj).success


def test_scienceqa_accepts_letter_labels():
    task = Task(
        task_id="q2",
        instruction="...",
        source="scienceqa",
        initial_observation=_obs(),
        gold_answer="chloroplast",
        meta={"choices": ["nucleus", "chloroplast", "mitochondrion"]},
    )
    traj = Trajectory(task_id="q2", final_answer="B")
    assert scienceqa_task_success(task, traj).success


def test_mind2web_requires_full_sequence():
    gold = [
        Action(type=ActionType.CLICK, target="#menu"),
        Action(type=ActionType.CLICK, target="#item"),
    ]
    task = Task(
        task_id="m1", instruction="...", source="mind2web",
        initial_observation=_obs(), gold_actions=gold, max_steps=5,
    )
    steps = [
        TrajectoryStep(observation=_obs(), action=Action(type=ActionType.CLICK, target="#menu")),
        TrajectoryStep(observation=_obs(), action=Action(type=ActionType.CLICK, target="#item")),
    ]
    traj = Trajectory(task_id="m1", steps=steps, terminated=True)
    assert mind2web_task_success(task, traj).success

    # Missing the second step -> not success, but progress == 0.5.
    traj_partial = Trajectory(task_id="m1", steps=steps[:1], terminated=True)
    score = mind2web_task_success(task, traj_partial)
    assert not score.success
    assert 0.4 < score.progress < 0.6


def test_synthetic_scoring_is_first_step_only():
    task = Task(
        task_id="s1", instruction="click login", source="synthetic",
        initial_observation=_obs(),
        gold_actions=[Action(type=ActionType.CLICK, target="#login-button")],
    )
    # Fuzzy match: "login button" vs "#login-button" should still pass.
    traj = Trajectory(
        task_id="s1",
        steps=[TrajectoryStep(observation=_obs(), action=Action(type=ActionType.CLICK, target="login button"))],
    )
    assert synthetic_task_success(task, traj).success
