"""Stored output hashes: a kernel or dependency change that moves any bit fails here. K31_GOLDEN_UPDATE=1 rewrites them."""

import hashlib
import json
import os
import pathlib

PATH = pathlib.Path(__file__).with_name("golden.json")


def digest(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().contiguous().cpu()
        h.update(str((t.dtype, tuple(t.shape))).encode())
        h.update(t.view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def check(name, *tensors):
    got = digest(*tensors)
    table = json.loads(PATH.read_text()) if PATH.exists() else {}
    if os.environ.get("K31_GOLDEN_UPDATE") == "1":
        table[name] = got
        PATH.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
        return
    assert name in table, f"no golden for {name}; run once with K31_GOLDEN_UPDATE=1"
    assert table[name] == got, f"{name}: output bits changed ({got[:16]} != {table[name][:16]})"
