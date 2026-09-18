"""Local-only Harbor agent wrapper that mutates task images for convenience."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from ridges_harbor.agents import RidgesMinerAgent

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment

LOCAL_RUNTIME_BOOTSTRAP_LOG_FILENAME = "local-runtime-bootstrap.log"
LOCAL_RUNTIME_BASELINE_REQUIREMENTS_PATH = Path(__file__).with_name("baseline-requirements.txt")
# Where `local_harbor.py` bind-mounts the host wheelhouse, when there is one.
# Nothing here creates the mount: this module only notices it at runtime, so a
# cell run without one behaves exactly as it did before.
WHEELHOUSE_DIR = "/wheels"
WHEELHOUSE_LOCK = WHEELHOUSE_DIR + "/requirements.lock"


def _read_requirements(path: Path) -> tuple[str, ...]:
    """Load one requirement per non-comment line from a requirements file."""
    requirements: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return tuple(requirements)


def _requirement_to_distribution_name(requirement: str) -> str:
    """Best-effort distribution name for a baseline requirement."""
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip()


LOCAL_RUNTIME_MINER_PACKAGES = _read_requirements(LOCAL_RUNTIME_BASELINE_REQUIREMENTS_PATH)
LOCAL_RUNTIME_DISTRIBUTION_NAMES = tuple(
    _requirement_to_distribution_name(package) for package in LOCAL_RUNTIME_MINER_PACKAGES
)


def _build_local_runtime_bootstrap_command() -> str:
    """Build the best-effort shell command that ensures local miner baseline deps."""
    packages = " ".join(LOCAL_RUNTIME_MINER_PACKAGES)
    probe = (
        "import importlib.metadata as metadata\n"
        "import sys\n"
        f"dists={list(LOCAL_RUNTIME_DISTRIBUTION_NAMES)!r}\n"
        "missing=[]\n"
        "for dist in dists:\n"
        "    try:\n"
        "        metadata.version(dist)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        missing.append(dist)\n"
        "sys.exit(0 if not missing else 1)\n"
    )
    # Some task images (e.g. buildpack-deps) ship python3 without pip or ensurepip,
    # so fall back to the distro package before giving up.
    ensure_pip = (
        "python3 -m pip --version >/dev/null 2>&1"
        " || python3 -m ensurepip --upgrade"
        " || (command -v apt-get >/dev/null 2>&1 &&"
        " apt-get update && apt-get install -y python3-pip)"
        " || (command -v apk >/dev/null 2>&1 && apk add --no-cache py3-pip)"
    )
    # Two ways to install the same 28 distributions, chosen by whether a
    # wheelhouse is mounted -- and the choice is made in the container, at run
    # time, so a cell launched without the mount is byte-for-byte the old
    # behaviour rather than a new path that has to be kept working.
    #
    # Why it matters: the network install pulls 62 wheels / 88 MB (numpy,
    # scipy, pandas, lxml, tree-sitter-language-pack) with `--no-cache-dir`,
    # once per CELL, not once per image.  At 12 cells in flight that is a
    # gigabyte of PyPI, and on 2026-09-03 four of twenty-eight cells died
    # in `AgentSetupTimeoutError` at 360s while doing it.  The bottleneck was
    # the link, not the cores.  It is also a reproducibility hole: the versions
    # float, so two cells of one arm can resolve different ones (regex-2026.9.3
    # was resolved at run time).  The wheelhouse fixes both -- the lock file
    # pins every version, and installing from it is a disk copy.
    install_online = (
        f"python3 -m pip install --no-cache-dir {packages}"
        f" || python3 -m pip install --break-system-packages --no-cache-dir {packages}"
    )
    offline_flags = f"--no-cache-dir --no-index --find-links {WHEELHOUSE_DIR} -r {WHEELHOUSE_LOCK}"
    install_offline = (
        f"python3 -m pip install {offline_flags}"
        f" || python3 -m pip install --break-system-packages {offline_flags}"
    )
    # The fallback is not belt-and-braces, it is the only thing that makes the
    # wheelhouse safe to mount everywhere.  A wheelhouse is built for ONE
    # interpreter ABI: the set28 packs are python:3.13-slim, but 7 of the 8
    # rebuilt set27 packs are ubuntu:24.04, whose python3 is 3.12.  Mounted
    # there, `--no-index` finds no compatible wheel and pip fails -- and the
    # bootstrap's own contract is to swallow that failure, so without this line
    # those cells would quietly run WITHOUT numpy/pandas/tree-sitter while
    # reporting nothing at all.  A cell that cannot use the wheelhouse must end
    # up exactly where it was before it existed.
    #
    # Written with `%` rather than an f-string on purpose: a literal `}` inside
    # an f-string is a SyntaxError before Python 3.12 accepted it under PEP 701,
    # and the two interpreters in play here are not the same one -- this file is
    # edited against 3.13 on the Mac and imported by ~/.venv-ridges (3.12) on
    # the eval box, where the first version of this line refused to import and
    # every cell died in four seconds.
    fallback = (
        "|| { echo '[ridges] wheelhouse does not fit this image; "
        "falling back to PyPI'; %s; }" % install_online
    )
    choose = (
        "if [ -f %s ]; then"
        " echo '[ridges] installing from wheelhouse %s (no network)';"
        " %s %s;"
        " else %s;"
        " fi"
        % (WHEELHOUSE_LOCK, WHEELHOUSE_DIR, install_offline, fallback, install_online)
    )
    # Best-effort by contract: the miner agent may well be stdlib-only, so a failure
    # here must not abort the run. The full transcript still lands in the log file.
    return (
        f"python3 -c {shlex.quote(probe)}"
        " || ("
        f"{ensure_pip}; "
        f"{choose}"
        ")"
        " || echo '[ridges] baseline package bootstrap failed; continuing'"
    )


class LocalMinerAgent(RidgesMinerAgent):
    """Local-testing agent that best-effort installs the common miner baseline."""

    async def _bootstrap_runtime_dependencies(self, environment: "BaseEnvironment") -> None:
        await super()._bootstrap_runtime_dependencies(environment)
        await self._exec_with_log(
            environment,
            executor=self.exec_as_root,
            command=_build_local_runtime_bootstrap_command(),
            log_filename=LOCAL_RUNTIME_BOOTSTRAP_LOG_FILENAME,
            cancelled_detail="command execution was cancelled",
            error_summary="Failed to install local miner baseline packages",
            error_type=RuntimeError,
        )
