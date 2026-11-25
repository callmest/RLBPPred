"""Check the installed package metadata, not just the documented pip command."""
from importlib.metadata import requires

import pytest
from packaging.requirements import Requirement


@pytest.mark.parametrize(("platform", "allowed"), [
    ("linux", "2.10.0+cpu"), ("win32", "2.10.0+cpu"), ("darwin", "2.10.0"),
])
def test_torch_distribution_pin(platform, allowed):
    requirements = [Requirement(s) for s in requires("rlbppred")]
    active = [r for r in requirements if r.name == "torch"
              and (r.marker is None or r.marker.evaluate({"sys_platform": platform}))]
    assert len(active) == 1
    for candidate in ("2.10.0", "2.10.0+cpu", "2.10.0+cu128", "2.10.0+rocm7.1",
                      "2.10.0+xpu", "2.9.1+cpu", "2.11.0+cpu"):
        assert (candidate in active[0].specifier) == (candidate == allowed)
