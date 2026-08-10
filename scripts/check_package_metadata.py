"""Validate installed package metadata used by the release workflow."""
from importlib.metadata import metadata

m = metadata("differentiable-physical-control-lab")
assert m["Name"] == "differentiable-physical-control-lab"
assert m["Version"] == "0.2.1"
assert m["Requires-Python"] == ">=3.11"
assert m["License-Expression"] == "MIT"
assert "LICENSE" in (m.get_all("License-File") or [])
assert m["Author"] == "Pratyush Kumar Das"
print("PASS: package metadata")
print("License-Expression:", m["License-Expression"])
print("License-File:", ", ".join(m.get_all("License-File") or []))
print("Requires-Python:", m["Requires-Python"])
