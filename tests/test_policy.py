"""Issue #21: the temperature and context budget environment overrides.

The defaults are the local H0 values. A cloud run exports both variables on
the host; the adapter forwards them into the sandbox, so a fresh import of
``harness.policy`` must see the override. Each case runs in a subprocess so a
test never mutates the shared module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

AGORA = Path(__file__).resolve().parents[1] / "domains" / "agora"
OVERRIDES = ("AGORA_TEMPERATURE", "AGORA_CONTEXT_CHARS")
READ = ("import json\n"
        "from harness import policy\n"
        "print(json.dumps([policy.TEMPERATURE, policy.CONTEXT_CHARS]))")


def _read(**overrides: str) -> list:
    env = {name: value for name, value in os.environ.items()
           if name not in OVERRIDES}
    env.update(overrides)
    result = subprocess.run([sys.executable, "-c", READ], cwd=AGORA, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_defaults_are_the_local_h0_values():
    assert _read() == [1.0, 48000]


def test_env_overrides_are_used():
    values = _read(AGORA_TEMPERATURE="0.6", AGORA_CONTEXT_CHARS="96000")
    assert values == [0.6, 96000]
