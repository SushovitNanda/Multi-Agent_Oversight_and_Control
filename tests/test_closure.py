from audit.closure import edges_from_peer_logs, effect_closure, knowledge_closure, message_closure, state_closure


def test_message_omission_is_flagged():
    logs = {"a": [], "b": [{"kind": "receive", "msg_id": "m1", "sender": "a", "content": "x"}]}
    assert message_closure(logs) == {"m1"}


def test_matching_message_passes():
    logs = {"a": [{"kind": "send", "msg_id": "m1", "receiver": "b", "content": "x"}],
            "b": [{"kind": "receive", "msg_id": "m1", "sender": "a", "content": "x"}]}
    assert message_closure(logs) == set()


def test_mutual_erasure_is_invisible():
    # Documents the audit's limit: if every party to a message lies, message closure cannot see it.
    assert message_closure({"a": [], "b": []}) == set()


def test_state_without_writer_is_flagged():
    assert state_closure([{"key": "f", "version": 2}], [{"key": "f", "version": 1}]) == [{"key": "f", "version": 2}]


def test_spoofed_call_is_flagged():
    claimed = [{"call_id": "c1", "name": "run_check", "arguments": "{}"}]
    effects = [{"call_id": "c1", "name": "write_file", "arguments": "{}"}]
    assert effect_closure(claimed, effects) == {"c1"}


def test_hidden_transfer_is_flagged():
    owners = {"PORT-7319": "a"}
    assert knowledge_closure(owners, [("c", "PORT-7319")], [("a", "b")]) == [("c", "PORT-7319")]


def test_logged_path_passes():
    owners = {"PORT-7319": "a"}
    edges = [("a", "b"), ("b", "c")]
    assert knowledge_closure(owners, [("c", "PORT-7319")], edges) == []


def test_time_order_respected():
    owners = {"PORT-7319": "a"}
    # c saw the tracer in round 2 but the only path a->b->c completes in round 5: flagged.
    edges = [("a", "b", 1), ("b", "c", 5)]
    assert knowledge_closure(owners, [("c", "PORT-7319", 2)], edges) == [("c", "PORT-7319")]
    assert knowledge_closure(owners, [("c", "PORT-7319", 6)], edges) == []


def test_out_of_order_hops_do_not_count():
    owners = {"PORT-7319": "a"}
    # b -> c happened before a -> b, so c cannot have got it that way.
    edges = [("b", "c", 1), ("a", "b", 3)]
    assert knowledge_closure(owners, [("c", "PORT-7319", 4)], edges) == [("c", "PORT-7319")]


def test_last_round_send_without_receive_is_not_flagged():
    logs = {"a": [{"round": 20, "kind": "send", "msg_id": "m9", "receiver": "b", "content": "bye"}], "b": []}
    assert message_closure(logs) == {"m9"}
    assert message_closure(logs, last_round=20) == set()
    assert message_closure(logs, last_round=21) == {"m9"}


def test_file_read_counts_as_logged_path():
    owners = {"APR-1111-AA": "a"}
    logs = {
        "a": [{"round": 1, "kind": "send", "msg_id": "m1", "receiver": "b", "content": "APR-1111-AA"}],
        "b": [{"round": 2, "kind": "receive", "msg_id": "m1", "sender": "a", "content": "APR-1111-AA"},
              {"round": 3, "kind": "state_write", "key": "fixes/x.md", "version": 1}],
        "c": [{"round": 5, "kind": "state_read", "key": "fixes/x.md", "version": 1}],
    }
    edges = edges_from_peer_logs(logs)
    assert ("b", "c", 5) in edges
    assert knowledge_closure(owners, [("c", "APR-1111-AA", 6)], edges) == []
    assert knowledge_closure(owners, [("c", "APR-1111-AA", 4)], edges) == [("c", "APR-1111-AA")]
