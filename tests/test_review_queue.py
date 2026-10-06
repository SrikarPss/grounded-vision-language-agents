from grounded_vla.synthetic.review import ReviewQueue, ReviewStatus


def test_two_person_approval_flow(tmp_path):
    q = ReviewQueue(state_path=tmp_path / "q.json", reviewers=("alice", "bob"))
    q.enqueue("t1")
    q.enqueue("t2")
    assert q.status("t1") == ReviewStatus.PENDING

    q.vote("t1", "alice", "approve")
    assert q.status("t1") == ReviewStatus.PENDING  # still needs bob

    q.vote("t1", "bob", "approve")
    assert q.status("t1") == ReviewStatus.APPROVED

    q.vote("t2", "alice", "approve")
    q.vote("t2", "bob", "reject")
    assert q.status("t2") == ReviewStatus.NEEDS_TIEBREAK

    assert q.approved_ids() == ["t1"]
    assert q.summary()["approved"] == 1


def test_review_state_persists(tmp_path):
    path = tmp_path / "q.json"
    q = ReviewQueue(state_path=path, reviewers=("a", "b"))
    q.enqueue("t1")
    q.vote("t1", "a", "approve")
    # Re-open:
    q2 = ReviewQueue(state_path=path, reviewers=("a", "b"))
    assert q2.status("t1") == ReviewStatus.PENDING
    q2.vote("t1", "b", "approve")
    assert q2.status("t1") == ReviewStatus.APPROVED
