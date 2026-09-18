"""Grade a separate-verifier task in its own container, the way the pack means it.

A set28 pack declares `[verifier] environment_mode = "separate"` and ships two
builds of the same repo: `environment/` for the agent and `tests/` for the
grader.  They are not the same image.  Only the grader's runs
`create_source_manifest.py` (which writes `/opt/task/SOURCE_REVISION` and the
untouched copy of the file under repair), only it carries `/tests`, and its
sidecars come up with different database passwords than the ones the agent is
given.  Its `test.sh` then applies `/logs/agent/patch.diff` to its own clean
`/app` and grades that.

harbor 0.3.0 has no notion of `environment_mode`: it runs `test.sh` inside the
agent's container.  That got the wrong answer twice over -- the patch was
applied a second time to a tree that already had it (`patch does not apply`,
every set28 cell 0 for two months), and even with the agent's apply removed the
honesty gates fail on a `/opt/task` that the agent image never had.  A shared
container cannot grade these packs; there is no flag for it.

So the verifier stage is run here instead, by the same steps the pack's own
grading script uses:

    build tests/Dockerfile  ->  compose up tests/docker-compose.yaml
    ->  run one container of that image on that network
    ->  drop the patch at /logs/agent/patch.diff  ->  bash /tests/test.sh
    ->  copy /logs/verifier back out  ->  tear the whole project down

Nothing here is a re-implementation of the grading contract: `test.sh` is the
pack's, and this file only puts it where it expects to be.  What it is
responsible for is that a run leaves no containers, networks, volumes or images
behind, and that every container it starts wears the bad-core fence.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# The healthy cpuset, supplied by bench/ops/lane.py through harbor.py.  Same
# name the fenced Docker environment reads, because it is the same fence: one
# core of the eval box corrupts what runs on it and says nothing, so the
# verifier is no more allowed onto it than the agent is.
CPUSET_ENV = "BENCH_CPUSET"

# The ch packs interpolate this into their compose file, so it has to be in the
# environment of the compose call rather than only on the container.
COMPOSE_CPUSET_VAR = "TASK_CPUSET"

CONTAINER_PATCH_PATH = "/logs/agent/patch.diff"
CONTAINER_VERIFIER_DIR = "/logs/verifier"


class SeparateVerifierError(RuntimeError):
    """The verifier environment could not be built, started, or read back."""


def _project_name(trial_name: str) -> str:
    """A compose project name that docker accepts and no other cell shares.

    Cells run concurrently and compose keys everything -- containers, network,
    volumes -- on this string, so two cells under one name would tear down each
    other's sidecars mid-grade.  The trial name is already unique per cell.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", trial_name.lower()).strip("-")
    return ("sn62v-" + slug)[:54].rstrip("-")


async def _run(command: list[str], *, timeout: float | None = None,
               env: dict[str, str] | None = None) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **(env or {})},
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise
    for line in (err or b"").decode("utf-8", "replace").splitlines():
        if line.strip():
            logger.warning("Separate verifier command stderr: %s", line)
    return process.returncode, (out or b"").decode("utf-8", "replace"), (err or b"").decode("utf-8", "replace")


class SeparateVerifier:
    """One task's verifier environment, from build to teardown."""

    def __init__(
        self,
        *,
        task_dir: Path,
        trial_dir: Path,
        trial_name: str,
        cpus: float | None = None,
        memory_mb: int | None = None,
        build_timeout_sec: float | None = None,
        timeout_sec: float | None = None,
    ):
        self.task_dir = Path(task_dir)
        self.trial_dir = Path(trial_dir)
        self.project = _project_name(trial_name)
        self.image = "%s:tests" % self.project
        self.container = "%s-verifier" % self.project
        self.cpus = cpus
        self.memory_mb = memory_mb
        self.build_timeout_sec = build_timeout_sec or 1800.0
        self.timeout_sec = timeout_sec or 900.0
        self.cpuset = (os.getenv(CPUSET_ENV) or "").strip()
        self.work = self.trial_dir / "verifier-environment"
        self.log = self.trial_dir / "separate-verifier.log"

    # -- plumbing ---------------------------------------------------------

    def _note(self, text: str) -> None:
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip("\n") + "\n")

    async def _docker(self, *args: str, timeout: float | None = None,
                      check: bool = True, env: dict[str, str] | None = None) -> str:
        code, out, err = await _run(["docker", *args], timeout=timeout, env=env)
        self._note("$ docker %s\n[return_code] %d\n%s" % (" ".join(args), code, out))
        if err:
            self._note("stderr: " + err[:4000])
        if check and code != 0:
            raise SeparateVerifierError(
                "docker %s failed (%d):\n%s" % (" ".join(args[:2]), code, out[-4000:])
            )
        return out

    @property
    def _compose_files(self) -> list[str]:
        files = ["-f", str(self.work / "docker-compose.yaml")]
        override = self.work / "cpuset.override.yaml"
        if override.is_file():
            files += ["-f", str(override)]
        return files

    def _compose_env(self) -> dict[str, str]:
        # Empty rather than absent: a compose file that interpolates the
        # variable would otherwise warn and substitute nothing, and the
        # difference between "no fence on this box" and "the fence silently
        # evaluated to blank" is the whole point of reading it back.
        return {COMPOSE_CPUSET_VAR: self.cpuset}

    def _limits(self) -> list[str]:
        flags: list[str] = []
        if self.cpuset:
            flags += ["--cpuset-cpus", self.cpuset]
        if self.cpus:
            flags += ["--cpus", str(self.cpus)]
        if self.memory_mb:
            flags += ["--memory", "%dm" % int(self.memory_mb)]
        return flags

    # -- stages -----------------------------------------------------------

    def _stage(self) -> None:
        """The pack's `tests/` as a build context, plus the cpuset override.

        Copied rather than built in place: the override file is written next to
        the compose file it overrides, and writing into the task directory
        would change the pack's own digest.
        """
        if self.work.exists():
            shutil.rmtree(self.work)
        shutil.copytree(self.task_dir / "tests", self.work)

    async def _write_cpuset_override(self) -> None:
        """A compose override pinning every service, or nothing when unfenced."""
        if not self.cpuset:
            return
        services = await self._docker(
            "compose", "-f", str(self.work / "docker-compose.yaml"),
            "config", "--services", timeout=120, env=self._compose_env())
        names = [line.strip() for line in services.splitlines() if line.strip()]
        if not names:
            raise SeparateVerifierError("`docker compose config --services` listed no services")
        for name in names:
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
                raise SeparateVerifierError("unexpected Compose service listing: %s" % name)
        body = "services:\n" + "".join(
            '  %s:\n    cpuset: "%s"\n' % (name, self.cpuset) for name in names)
        (self.work / "cpuset.override.yaml").write_text(body, encoding="utf-8")

    async def _up(self) -> str:
        await self._docker(
            "compose", "-p", self.project, *self._compose_files,
            "up", "-d", "--build", "--wait",
            timeout=self.build_timeout_sec, env=self._compose_env())
        return "%s_default" % self.project

    async def _grade(self, patch: Path, network: str) -> None:
        await self._docker(
            "run", "-d", "--name", self.container, "--network", network,
            *self._limits(), self.image, "sh", "-c", "sleep infinity",
            timeout=300)
        await self._docker("exec", self.container, "mkdir", "-p",
                           "/logs/agent", "/logs/artifacts", CONTAINER_VERIFIER_DIR,
                           timeout=120)
        await self._docker("cp", str(patch),
                           "%s:%s" % (self.container, CONTAINER_PATCH_PATH),
                           timeout=300)
        # `test.sh` grades by writing reward.txt; its exit code is not the
        # verdict (harbor's own contract), so a non-zero rc is recorded and
        # the reward is still read.
        await self._docker("exec", self.container, "bash", "/tests/test.sh",
                           timeout=self.timeout_sec, check=False)

    async def _collect(self) -> float | None:
        out = self.trial_dir / "verifier"
        out.mkdir(parents=True, exist_ok=True)
        await self._docker("cp", "%s:%s/." % (self.container, CONTAINER_VERIFIER_DIR),
                           str(out), timeout=300, check=False)
        reward = out / "reward.txt"
        if not reward.is_file():
            return None
        try:
            return float(reward.read_text().strip())
        except ValueError:
            return None

    async def _down(self) -> None:
        """Leave the box exactly as it was found.

        By name, never by prune: this box is shared with another subnet whose
        images a `docker system prune -a` has taken out before.
        """
        await self._docker("rm", "-f", self.container, timeout=120, check=False)
        if (self.work / "docker-compose.yaml").is_file():
            await self._docker(
                "compose", "-p", self.project, *self._compose_files,
                "down", "-v", "--rmi", "local", "--remove-orphans",
                timeout=600, check=False, env=self._compose_env())
        await self._docker("rmi", "-f", self.image, timeout=300, check=False)

    # -- entry point ------------------------------------------------------

    async def grade(self, patch: Path) -> float | None:
        """Build, grade, collect, tear down.  Returns the reward, or None."""
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        self._stage()
        try:
            await self._docker(
                "build", "-t", self.image, "-f", str(self.work / "Dockerfile"),
                str(self.work), timeout=self.build_timeout_sec)
            await self._write_cpuset_override()
            network = await self._up()
            await self._grade(patch, network)
            return await self._collect()
        finally:
            try:
                await self._down()
            except Exception as exception:                      # noqa: BLE001
                logger.warning("separate verifier teardown failed: %s", exception)


async def grade_with_separate_verifier(
    *,
    task_dir: Path,
    trial_dir: Path,
    trial_name: str,
) -> float | None:
    """Run the pack's own verifier for one finished trial.

    Returns None when the agent produced no patch at all -- there is nothing to
    grade, and a 0 written here would be indistinguishable from a graded
    failure.
    """
    import tomllib

    patch = Path(trial_dir) / "agent" / "patch.diff"
    if not patch.is_file():
        return None
    config = tomllib.loads((Path(task_dir) / "task.toml").read_text(encoding="utf-8"))
    verifier = config.get("verifier") or {}
    environment = verifier.get("environment") or {}
    runner = SeparateVerifier(
        task_dir=Path(task_dir),
        trial_dir=Path(trial_dir),
        trial_name=trial_name,
        cpus=environment.get("cpus"),
        memory_mb=environment.get("memory_mb"),
        build_timeout_sec=environment.get("build_timeout_sec"),
        timeout_sec=verifier.get("timeout_sec"),
    )
    return await verifier_result(runner, patch)


async def verifier_result(runner: SeparateVerifier, patch: Path) -> float | None:
    """One place for the try/except, so a broken environment is not a zero.

    A verifier that never came up has said nothing about the patch.  Harbor's
    own readers treat a missing reward as "this cell has to run again" and a 0
    as the agent's answer, and the difference between those two is the whole
    honesty of the local ruler.
    """
    try:
        return await runner.grade(patch)
    except (SeparateVerifierError, asyncio.TimeoutError, OSError) as exception:
        logger.warning("separate verifier failed for %s: %s",
                       runner.project, exception)
        runner._note("[failed] %s" % exception)
        return None
