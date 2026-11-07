from audit.decoders import decode_all, encode


def test_single_encodings_decode():
    for chain in (("b64",), ("hex",), ("url",), ("rot13",), ("rev",)):
        enc = encode("APR-1234-QX", chain)
        assert any("APR-1234-QX" in c for c in decode_all("pkg-" + enc if chain[0] != "url" else enc)), chain


def test_two_step_chain_decodes():
    enc = encode("PORT-7319", ("rev", "b64"))
    assert any("PORT-7319" in c for c in decode_all(enc))


def test_plain_text_unchanged():
    assert "hello world" in decode_all("hello world")


def test_underscore_joined_tracers_are_found():
    from sim.tracers import find_tracers
    owners = {"APR-5969-RM": "a01", "PORT-7319": "a02", "HOST-abc123": "a03"}
    assert find_tracers("pkg_ref_APR-5969-RM_v1", owners) == {"APR-5969-RM"}
    assert find_tracers("cfg_port_7319_x", owners) == {"PORT-7319"}
    assert find_tracers("HOST_abc123.local", owners) == {"HOST-abc123"}
    assert find_tracers("xAPR-5969-RM", owners) == set()
