"""Issue #21: the temperature and context budget environment overrides.

The defaults are the local H0 values. A cloud run exports the variables on the
host; the adapter forwards them into the sandbox, so a fresh import of
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
OVERRIDES = ("AGORA_TEMPERATURE", "AGORA_CONTEXT_WINDOW",
             "AGORA_SUMMARY_MAX_TOKENS")
READ = ("import json\n"
        "from harness import policy\n"
        "print(json.dumps([policy.TEMPERATURE, policy.CONTEXT_WINDOW,"
        " policy.CONTEXT_TOKENS, policy.KEEP_RECENT_TOKENS,"
        " policy.SUMMARY_MAX_TOKENS]))")


def _read(**overrides: str) -> list:
    env = {name: value for name, value in os.environ.items()
           if name not in OVERRIDES}
    env.update(overrides)
    result = subprocess.run([sys.executable, "-c", READ], cwd=AGORA, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_defaults_are_0_6_and_the_local_window():
    assert _read() == [0.6, 262144, 222822, 20000, 16384]


def test_env_overrides_are_used():
    values = _read(AGORA_TEMPERATURE="0.3", AGORA_CONTEXT_WINDOW="131072")
    # 85% of 131072, and the keep/summary defaults are unchanged.
    assert values == [0.3, 131072, 111411, 20000, 16384]


def test_summary_cap_override():
    assert _read(AGORA_SUMMARY_MAX_TOKENS="8192")[4] == 8192
