import responses_replay_ids


def test_function_item_id_preserves_short_ids():
    assert responses_replay_ids.function_item_id("call_123") == "fc_call_123"


def test_function_item_id_hashes_long_call_ids_within_upstream_limit():
    call_id = "call_" + "x" * 200

    item_id = responses_replay_ids.function_item_id(call_id)

    assert len(item_id) <= 64
    assert item_id == responses_replay_ids.function_item_id(call_id)
    assert item_id.startswith("fc_")


def test_repair_missing_replay_ids_rewrites_oversized_function_ids():
    call_id = "call_" + "x" * 200
    body = {
        "prompt_cache_key": "excel-thread",
        "input": [
            {
                "type": "function_call_output",
                "id": "fc_" + call_id,
                "call_id": call_id,
                "output": "done",
            }
        ],
    }

    repaired, trace = responses_replay_ids.repair_missing_replay_ids(body)

    assert len(repaired["input"][0]["id"]) <= 64
    assert repaired["input"][0]["id"] == responses_replay_ids.function_item_id(call_id)
    assert trace["input_items"] == 1
    assert trace["repaired_items"] == 1
    assert trace["repaired_by_type"] == {"function_call_output": 1}
    assert trace["lineage_key_kind"] == "prompt_cache"
    assert isinstance(trace["lineage_key_sha256"], str)
