"""Unit tests for the Unikraft Cloud environment."""

from __future__ import annotations

import base64
import io
import json
import logging
import shlex
import shutil
import subprocess
import tarfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("unikraft_cloud")

from unikraft_cloud import UnikraftCloudError
from unikraft_cloud.core.http import ApiClientConfig
from unikraft_cloud.plugins.sandbox import ExecResult as SandboxExecResult
from unikraft_cloud.plugins.sandbox import ExecTimeoutError, OutputChunk

import harbor.environments.unikraft as unikraft_module
from harbor.environments.factory import _ENVIRONMENT_REGISTRY
from harbor.environments.tar_transfer import pack_dir_to_bytes
from harbor.environments.unikraft import (
    UnikraftEnvironment,
    _image_repository,
    _run_as_user,
    _run_in_shell,
    _sanitize_name,
    _ShieldApi,
    parse_dockerfile_env,
    parse_dockerfile_user,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import TrialPaths

SELECT_SHELL = "_shell=$(command -v bash || echo /bin/sh)"


def inner_command(cmd: str) -> str:
    """The command the guest shell wrapper runs."""
    return shlex.split(cmd)[-1]


Answer = tuple[int, bytes, bytes]


class FakeCommand:
    def __init__(self, uuid: str, answer: Answer, chunks: list[OutputChunk]) -> None:
        self.uuid = uuid
        self.exit_code, self.stdout, self.stderr = answer
        self.chunks = chunks
        self.signals: list[int | str] = []
        self.deleted = False

    async def stream(self) -> AsyncIterator[OutputChunk]:
        for chunk in self.chunks:
            yield chunk

    async def get(self) -> SimpleNamespace:
        return SimpleNamespace(exitcode=self.exit_code)

    async def signal(self, signal: int | str) -> None:
        self.signals.append(signal)

    async def delete(self) -> None:
        self.deleted = True


class FakeFiles:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = set()
        self.calls: list[tuple[Any, ...]] = []

    @staticmethod
    def _bytes(data: bytes | str) -> bytes:
        return data.encode() if isinstance(data, str) else data

    async def upload(
        self, path: str, filename: str, data: bytes | str, *, parents: bool = False
    ) -> None:
        # The plugin writes to the given path itself unless it is a directory,
        # in which case the file lands under it.
        base = path.rstrip("/") or "/"
        if parents:
            self.dirs.update(str(item) for item in PurePosixPath(base).parents)
        target = f"{base}/{filename}" if base in self.dirs else base
        self.files[target] = self._bytes(data)
        self.calls.append(("upload", path, filename, len(data), parents))

    async def write(
        self, path: str, data: bytes | str, *, append: bool = False
    ) -> None:
        previous = self.files.get(path, b"") if append else b""
        self.files[path] = previous + self._bytes(data)
        self.calls.append(("write", path, len(data), append))

    async def read(self, path: str) -> bytes:
        return self.files[path]


class FakeSandbox:
    """The sandbox plugin client, answering commands from a script."""

    def __init__(self) -> None:
        self.fs = FakeFiles()
        self.ready_timeouts: list[float] = []
        self.execs: list[dict[str, Any]] = []
        self.commands: dict[str, FakeCommand] = {}
        self.script: Callable[[str], Answer] | None = None
        self.stream_chunks: list[OutputChunk] | None = None
        self.interrupted = False
        self.raise_timeout = False

    def _answer(self, cmd: str) -> Answer:
        return self.script(cmd) if self.script else (0, b"", b"")

    async def wait_ready(self, timeout: float = 60.0) -> None:
        self.ready_timeouts.append(timeout)

    async def run(
        self, cmd: str, *, cwd: str | None = None, env: dict[str, str] | None = None
    ) -> FakeCommand:
        uuid = f"cmd-{len(self.commands) + 1}"
        answer = self._answer(cmd)
        if self.stream_chunks is not None:
            chunks = list(self.stream_chunks)
        else:
            chunks = [OutputChunk("stdout", answer[1])] if answer[1] else []
            chunks += [OutputChunk("stderr", answer[2])] if answer[2] else []
        command = FakeCommand(uuid, answer, chunks)
        self.commands[uuid] = command
        self.execs.append({"cmd": cmd, "cwd": cwd, "env": env, "uuid": uuid})
        return command

    async def exec(
        self,
        cmd: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdin: bytes | str | None = None,
        timeout: float | None = None,
        wait_delay: float | None = None,
    ) -> SandboxExecResult:
        command = await self.run(cmd, cwd=cwd, env=env)
        self.execs[-1].update(timeout=timeout, wait_delay=wait_delay)
        if self.raise_timeout:
            raise ExecTimeoutError(command)  # type: ignore[arg-type]
        return SandboxExecResult(
            uuid=command.uuid,
            exit_code=command.exit_code,
            stdout=command.stdout,
            stderr=command.stderr,
            interrupted=self.interrupted,
        )

    def command(self, uuid: str) -> FakeCommand:
        return self.commands[uuid]


class FakePlugin:
    def __init__(self, client: FakeClient, uuid: str, name: str) -> None:
        self._client = client
        self._route = (
            f"https://api.test.unikraft.cloud/v1/instances/{uuid}/plugins/{name}"
        )

    async def client(self, factory: Callable[[ApiClientConfig], Any]) -> Any:
        config = ApiClientConfig(
            base_url=self._route, token="tok", http=self._client.shield_http
        )
        return factory(config)


class FakeHandle:
    def __init__(self, client: FakeClient, uuid: str) -> None:
        self._client = client
        self.uuid = uuid

    def sandbox(self, *, plugin: str = "sandbox") -> FakeSandbox:
        return self._client.sandbox

    def plugin(self, name: str) -> FakePlugin:
        return FakePlugin(self._client, self.uuid, name)

    async def delete(self) -> None:
        self._client.events.append(("delete", self.uuid))
        if self._client.fail_delete_once:
            self._client.fail_delete_once = False
            raise UnikraftCloudError("instance is running", kind="network")

    async def stop(self) -> None:
        self._client.events.append(("stop", self.uuid))

    async def suspend(self) -> None:
        self._client.events.append(("suspend", self.uuid))

    async def start(self) -> None:
        self._client.events.append(("start", self.uuid))

    def __await__(self) -> Any:
        async def read() -> SimpleNamespace:
            reason, code = self._client.stopped.get(self.uuid, (None, None))
            return SimpleNamespace(
                uuid=self.uuid,
                stop_reason=reason,
                stop_code=code,
                # The platform names the interface it gives an instance.
                network_interfaces=[SimpleNamespace(uuid=f"iface-{self.uuid}")],
            )

        return read().__await__()


class FakeInstances:
    def __init__(self, client: FakeClient) -> None:
        self._client = client

    async def create(self, **spec: Any) -> SimpleNamespace:
        self._client.created.append(spec)
        if self._client.create_failures > 0:
            self._client.create_failures -= 1
            uuid = f"stopped-{len(self._client.created)}"
            self._client.stopped[uuid] = self._client.stop_detail
            raise UnikraftCloudError(
                "Unikraft Cloud API reported an error",
                kind="api",
                errors=(SimpleNamespace(uuid=uuid),),
            )
        uuid = f"inst-{len(self._client.created)}"
        return SimpleNamespace(uuid=uuid, name=spec.get("name"), state="running")

    def get(self, *, uuid: str | None = None, name: str | None = None) -> FakeHandle:
        return FakeHandle(self._client, uuid or f"name:{name}")


#: The credentials a UKC token encodes, as the registry wants them.
REGISTRY_USER = "demo"
REGISTRY_TOKEN = base64.b64encode(b"demo:secret").decode()


class FakeRegistry:
    """The OCI registry: a token endpoint, and a manifest per image it holds."""

    def __init__(self) -> None:
        self.tags: set[str] = set()
        self.manifest_calls = 0
        self.token_calls = 0
        self.scopes: list[str] = []
        self.status: int | None = None

    def add(self, image_ref: str) -> None:
        self.tags.add(image_ref)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/service/token":
            self.token_calls += 1
            self.scopes.append(request.url.params.get("scope", ""))
            return httpx.Response(200, json={"token": "registry-token"})
        self.manifest_calls += 1
        if self.status is not None:
            return httpx.Response(self.status)
        if request.headers.get("authorization") != "Bearer registry-token":
            return httpx.Response(
                401,
                headers={
                    "www-authenticate": (
                        'Bearer realm="https://unikraft.io/service/token",'
                        'service="harbor-registry"'
                    )
                },
            )
        repository, _, tag = request.url.path.partition("/manifests/")
        reference = f"{repository.removeprefix('/v2/')}:{tag}"
        return httpx.Response(200 if reference in self.tags else 404)


class FakeClient:
    """The SDK client: instances, images and one sandbox behind them."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.instances = FakeInstances(self)
        self.registry = FakeRegistry()
        self.sandbox = FakeSandbox()
        self.created: list[dict[str, Any]] = []
        self.events: list[tuple[str, str]] = []
        self.fail_delete_once = False
        self.create_failures = 0
        self.stopped: dict[str, tuple[int | None, int | None]] = {}
        #: What the platform reports for an instance a failed create left behind.
        self.stop_detail: tuple[int | None, int | None] = (None, None)
        self.closed = False
        self.shield_requests: list[tuple[str, str, Any]] = []
        self.shield_failures = 0
        self.shield_http = httpx.AsyncClient(
            transport=httpx.MockTransport(self._shield)
        )

    def _shield(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.rsplit("/api/v1", 1)[-1]
        body = json.loads(request.content) if request.content else None
        self.shield_requests.append((request.method, path, body))
        if path == "/status":
            if self.shield_failures > 0:
                self.shield_failures -= 1
                return httpx.Response(502, text="plugin starting")
            return httpx.Response(200, json={"uptime_seconds": 1.0, "policy_count": 0})
        if path == "/policies" and request.method == "PUT":
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    async def aclose(self) -> None:
        self.closed = True
        await self.shield_http.aclose()


@pytest.fixture
def fake_ukc(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = FakeClient()

    def factory(**kwargs: Any) -> FakeClient:
        client.kwargs = kwargs
        return client

    monkeypatch.setattr(unikraft_module, "UnikraftCloud", factory)
    for variable in ("UKC_METRO", "UKC_USER"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("UKC_TOKEN", REGISTRY_TOKEN)

    real_client = httpx.AsyncClient

    def http_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        # A caller with its own transport, such as the shield, keeps it.
        kwargs.setdefault("transport", httpx.MockTransport(client.registry.handle))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(unikraft_module.httpx, "AsyncClient", http_factory)
    monkeypatch.setattr(unikraft_module, "_READY_FIRST_INTERVAL_SEC", 0.001)
    monkeypatch.setattr(unikraft_module, "_READY_MAX_INTERVAL_SEC", 0.001)
    return client


@pytest.fixture
def fake_build(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record every build as (Dockerfile contents, image reference)."""
    builds: list[tuple[str, str]] = []

    async def run_build(
        self: UnikraftEnvironment, dockerfile: Path, image_ref: str
    ) -> None:
        builds.append((dockerfile.read_text(), image_ref))

    monkeypatch.setattr(UnikraftEnvironment, "_run_build", run_build)
    return builds


def _make_env(
    temp_dir: Path,
    *,
    config: EnvironmentConfig | None = None,
    network_policy: NetworkPolicy | None = None,
    dockerfile: bool = True,
    dockerfile_contents: str = "FROM ubuntu:24.04\n",
    compose: bool = False,
    environment_name: str = "hello-world",
    **kwargs: Any,
) -> UnikraftEnvironment:
    environment_dir = temp_dir / "environment"
    environment_dir.mkdir(exist_ok=True)
    if dockerfile:
        (environment_dir / "Dockerfile").write_text(dockerfile_contents)
    if compose:
        (environment_dir / "docker-compose.yaml").write_text(
            "services:\n  main:\n    build: .\n"
        )
    trial_paths = TrialPaths(trial_dir=temp_dir / "trial")
    trial_paths.mkdir()
    kwargs.setdefault("image_namespace", "demo")
    kwargs.setdefault("metro", "https://api.test.unikraft.cloud/v1")
    return UnikraftEnvironment(
        environment_dir=environment_dir,
        environment_name=environment_name,
        session_id="hello-world__bZZeEkw__env",
        trial_paths=trial_paths,
        task_env_config=config or EnvironmentConfig(),
        network_policy=network_policy,
        **kwargs,
    )


def _allowlist(*hosts: str) -> NetworkPolicy:
    return NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=list(hosts))


def _shield_puts(client: FakeClient) -> list[Any]:
    return [body for method, path, body in client.shield_requests if method == "PUT"]


# ---------- registration and validation ----------


def test_factory_registers_unikraft() -> None:
    entry = _ENVIRONMENT_REGISTRY[EnvironmentType.UNIKRAFT]
    assert entry.module == "harbor.environments.unikraft"
    assert entry.class_name == "UnikraftEnvironment"
    assert entry.pip_extra == "unikraft"


def test_type_and_provider_name(fake_ukc: FakeClient, temp_dir: Path) -> None:
    env = _make_env(temp_dir)
    assert env.type() == EnvironmentType.UNIKRAFT
    assert env.provider_name == "unikraft"
    assert env.get_sandbox_id() is None


def test_capabilities_cover_shielded_network_modes(
    fake_ukc: FakeClient, temp_dir: Path
) -> None:
    capabilities = _make_env(temp_dir).capabilities
    assert capabilities.disable_internet is True
    assert capabilities.network_allowlist is True
    assert capabilities.network_allowlist_hostnames is True
    assert capabilities.network_allowlist_wildcard_hostnames is True
    assert capabilities.dynamic_network_policy is True
    assert capabilities.network_allowlist_ipv4_addresses is False
    assert capabilities.network_allowlist_ipv4_cidrs is False
    assert capabilities.gpus is False
    assert capabilities.docker_compose is False
    assert capabilities.windows is False


def test_resources_are_hard_limits(fake_ukc: FakeClient, temp_dir: Path) -> None:
    capabilities = UnikraftEnvironment.resource_capabilities()
    assert capabilities.cpu_limit is True
    assert capabilities.memory_limit is True
    assert capabilities.cpu_request is False
    assert capabilities.memory_request is False

    _make_env(
        temp_dir,
        config=EnvironmentConfig(cpus=2, memory_mb=1024),
        cpu_enforcement_policy=ResourceMode.LIMIT,
    )
    with pytest.raises(ValueError, match="memory resource requests"):
        _make_env(
            temp_dir,
            config=EnvironmentConfig(cpus=2, memory_mb=1024),
            memory_enforcement_policy=ResourceMode.REQUEST,
        )


def test_requires_an_environment_definition(
    fake_ukc: FakeClient, temp_dir: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="no environment definition"):
        _make_env(temp_dir, dockerfile=False)


def test_rejects_compose_tasks(fake_ukc: FakeClient, temp_dir: Path) -> None:
    with pytest.raises(ValueError, match="Docker Compose"):
        _make_env(temp_dir, compose=True)


def test_ip_allowlist_entries_are_rejected(
    fake_ukc: FakeClient, temp_dir: Path
) -> None:
    with pytest.raises(ValueError, match="IPv4 addresses is not supported"):
        _make_env(temp_dir, network_policy=_allowlist("192.0.2.1"))
    with pytest.raises(ValueError, match="IPv4 CIDR ranges is not supported"):
        _make_env(temp_dir, network_policy=_allowlist("192.0.2.0/24"))


def test_rejects_unusable_kwargs(fake_ukc: FakeClient, temp_dir: Path) -> None:
    with pytest.raises(ValueError, match="shield_handler"):
        _make_env(temp_dir, shield_handler="ftp")
    with pytest.raises(ValueError, match="shield_ports"):
        _make_env(temp_dir, shield_ports=[])
    with pytest.raises(ValueError, match="positive"):
        _make_env(temp_dir, default_memory_mb=0)


# ---------- images ----------


def test_image_ref_names_namespace_task_and_content_hash(
    fake_ukc: FakeClient, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _make_env(temp_dir, environment_name="Org/My Task")
    # Unqualified: the CLI resolves the registry against the metro's index.
    namespace, _, rest = env._image_ref().partition("/")
    name, _, tag = rest.partition(":")
    assert namespace == "demo"
    assert name == "harbor-org-my-task"
    assert tag == env.environment_id[:12]

    monkeypatch.setenv("UKC_USER", "alice")
    assert _make_env(temp_dir, image_namespace=None)._image_ref().startswith("alice/")

    # A registry pins the push destination instead.
    pinned = _make_env(temp_dir, registry="index.fra0-fe-test.unikraft.cloud")
    assert pinned._image_ref().startswith("index.fra0-fe-test.unikraft.cloud/demo/")

    monkeypatch.delenv("UKC_USER")
    with pytest.raises(ValueError, match="UKC_USER"):
        _make_env(temp_dir, image_namespace=None)._image_ref()


async def test_start_builds_only_a_missing_image(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir, dockerfile_contents="FROM alpine:3.20\n")
    ref = env._image_ref()
    fake_ukc.registry.add(ref)
    # Another task's image, and the same task at another revision, are not it.
    fake_ukc.registry.add("demo/harbor-other:7140400b7907")
    fake_ukc.registry.add(f"{ref.rsplit(':', 1)[0]}:0000deadbeef")

    await env.start(force_build=False)
    assert fake_ukc.registry.manifest_calls == 2  # the challenge, then the answer
    assert fake_ukc.registry.scopes == [f"repository:{_image_repository(ref)}:pull"]
    assert fake_build == []

    await env.start(force_build=True)
    assert fake_build == [("FROM alpine:3.20\n", ref)]


async def test_start_builds_when_the_image_is_absent_or_unknown(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir)
    await env.start(force_build=False)
    assert [ref for _, ref in fake_build] == [env._image_ref()]

    fake_ukc.registry.status = 500
    await env.start(force_build=False)
    assert len(fake_build) == 2


async def test_require_prebuilt_image_starts_from_an_existing_image(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir, require_prebuilt_image=True)
    fake_ukc.registry.add(env._image_ref())

    await env.start(force_build=False)
    assert fake_build == []
    assert fake_ukc.created


async def test_require_prebuilt_image_rejects_a_missing_image(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir, require_prebuilt_image=True)

    with pytest.raises(RuntimeError, match="does not exist"):
        await env.start(force_build=False)
    assert fake_build == []
    assert not fake_ukc.created


async def test_require_prebuilt_image_rejects_force_build(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir, require_prebuilt_image=True)
    fake_ukc.registry.add(env._image_ref())

    with pytest.raises(ValueError, match="force_build"):
        await env.start(force_build=True)
    assert fake_build == []


async def test_prebuilt_docker_image_is_wrapped_in_a_dockerfile(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(
        temp_dir,
        dockerfile=False,
        config=EnvironmentConfig(docker_image="ghcr.io/org/task:1.2"),
    )
    await env.start(force_build=False)
    assert fake_build == [("FROM ghcr.io/org/task:1.2\n", env._image_ref())]


# ---------- the instance ----------


async def test_instance_spec_maps_task_resources(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(
        temp_dir,
        config=EnvironmentConfig(cpus=2, memory_mb=4096, env={"TASK": "1"}),
        persistent_env={"TRIAL": "2"},
        ready_timeout_sec=90,
        autokill_after_stop_sec=600,
    )
    env.context_id = "594025f3-7d65-4655-8576-4bee95002eae"  # type: ignore[assignment]
    await env.start(force_build=False)

    assert fake_ukc.kwargs == {"metro": "https://api.test.unikraft.cloud/v1"}
    (spec,) = fake_ukc.created
    assert spec["image"] == env._image_ref()
    assert spec["memory_mb"] == 4096
    assert spec["vcpus"] == 2
    assert spec["args"] == ["/bin/sh", "-c", "sleep infinity"]
    assert spec["env"] == {"TASK": "1", "TRIAL": "2"}
    assert spec["plugins"] == [
        {"name": "sandbox", "image": "plugins/sandbox:staging", "config": {}}
    ]
    assert spec["restart_policy"] == "never"
    assert spec["autokill"] == {"time_ms": 600_000}
    assert spec["autostart"] is True
    assert spec["timeout_s"] == 60
    assert "network_interfaces" not in spec
    assert "scale_to_zero" not in spec
    assert spec["tags"] == [
        "harbor",
        "harbor.session=hello-world__bZZeEkw__env",
        "harbor.context=594025f3-7d65-4655-8576-4bee95002eae",
    ]
    assert spec["name"].startswith("hello-world-bzzeekw-env-")
    assert len(spec["name"]) <= 63
    assert fake_ukc.sandbox.ready_timeouts == [90]
    assert env.get_sandbox_id() == "inst-1"


async def test_instance_spec_falls_back_to_defaults(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(
        temp_dir,
        metro="https://api.fra0-fe-test.unikraft.cloud/v1",
        default_memory_mb=512,
        default_vcpus=4,
        keepalive=["/usr/bin/tail", "-f", "/dev/null"],
        plugin_image="plugins/sandbox:latest",
        ready_timeout_sec=30,
    )
    await env.start(force_build=False)

    assert fake_ukc.kwargs == {"metro": "https://api.fra0-fe-test.unikraft.cloud/v1"}
    (spec,) = fake_ukc.created
    assert spec["memory_mb"] == 512
    assert spec["vcpus"] == 4
    assert spec["args"] == ["/usr/bin/tail", "-f", "/dev/null"]
    assert spec["plugins"][0]["image"] == "plugins/sandbox:latest"
    assert spec["timeout_s"] == 30
    assert "env" not in spec


async def test_start_requires_a_metro(
    fake_ukc: FakeClient,
    fake_build: list[tuple[str, str]],
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_env(temp_dir, metro=None)
    with pytest.raises(ValueError, match="UKC_METRO"):
        await env.start(force_build=False)
    assert fake_ukc.created == []

    monkeypatch.setenv("UKC_METRO", "fra0-fe-test")
    await _make_env(temp_dir, metro=None).start(force_build=False)
    assert fake_ukc.kwargs == {"metro": "fra0-fe-test"}
    assert len(fake_ukc.created) == 1


def test_instance_specs_use_only_declared_sdk_fields(
    fake_ukc: FakeClient, temp_dir: Path
) -> None:
    from unikraft_cloud.api.platform import models

    from unikraft_cloud.core.resource import check_spec

    env = _make_env(temp_dir, network_policy=_allowlist("example.com"))
    env._shield_uuid = "shield"
    for spec in (env._instance_spec("demo/harbor-hello-world:abc"), env._shield_spec()):
        check_spec(spec, models.CreateInstanceRequest, "instance")
        request = models.CreateInstanceRequest.model_validate(spec)
        assert not request.model_extra, request.model_extra
        for interface in request.network_interfaces or []:
            assert not interface.model_extra, interface.model_extra
            if interface.relay is not None:
                assert not interface.relay.model_extra, interface.relay.model_extra
        for plugin in request.plugins or []:
            # The API renamed `rom` to `image`; the published spec still lags.
            assert set(plugin.model_extra or {}) == {"image"}
        assert request.autokill is not None and not request.autokill.model_extra


async def test_storage_mb_is_ignored_with_a_warning(
    fake_ukc: FakeClient,
    fake_build: list[tuple[str, str]],
    temp_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test-unikraft")
    env = _make_env(temp_dir, config=EnvironmentConfig(storage_mb=10240), logger=logger)
    with caplog.at_level(logging.WARNING, logger="test-unikraft"):
        await env.start(force_build=False)
    assert "storage_mb=10240" in caplog.text
    assert "volumes" not in fake_ukc.created[0]


# ---------- the network shield ----------


async def test_public_task_starts_without_a_shield(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    await _make_env(temp_dir).start(force_build=False)
    assert len(fake_ukc.created) == 1
    assert fake_ukc.shield_requests == []


async def test_allowlist_task_provisions_a_shield_first(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir, network_policy=_allowlist("api.github.com", "*.pypi.org"))
    await env.start(force_build=False)

    shield, instance = fake_ukc.created
    assert shield["image"] == "demo/netshield:latest"
    assert shield["memory_mb"] == 1024
    assert shield["plugins"] == [
        {"name": "netshield", "image": "demo/netshield-api:latest", "config": {}}
    ]
    # The shield names no interface, so a failed attempt leaves nothing to collide
    # with; the task relays through whichever one the platform gave it.
    assert "network_interfaces" not in shield
    assert instance["network_interfaces"] == [
        {"relay": {"uuid": f"iface-{env._shield_uuid}", "relay_dns": True}}
    ]
    assert shield["name"] == instance["name"] + "-shield"
    assert "service_group" not in shield

    methods = [method for method, _, _ in fake_ukc.shield_requests]
    assert methods.index("GET") < methods.index("PUT")
    assert _shield_puts(fake_ukc) == [
        [
            {
                "id": "harbor-egress",
                "direction": "outbound",
                "handler": "passthrough",
                "match": {
                    "hosts": ["api.github.com", "*.pypi.org"],
                    "ports": [80, 443],
                },
                "priority": 100,
            }
        ]
    ]
    assert env._shield_uuid == "inst-1"
    assert env.get_sandbox_id() == "inst-2"


async def test_no_network_task_installs_an_empty_policy_set(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(
        temp_dir, network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    )
    await env.start(force_build=False)
    assert len(fake_ukc.created) == 2
    assert _shield_puts(fake_ukc) == [[]]


async def test_phase_policy_forces_a_shield_with_public_passthrough(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(
        temp_dir,
        network_policy=NetworkPolicy(),
        phase_network_policies=[_allowlist("example.com")],
    )
    await env.start(force_build=False)
    assert len(fake_ukc.created) == 2
    assert _shield_puts(fake_ukc)[0][0]["match"]["hosts"] == ["*"]


async def test_set_network_policy_replaces_the_shield_policies(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir, network_policy=_allowlist("example.com"))
    await env.start(force_build=False)

    await env.set_network_policy(_allowlist("example.com"))
    assert len(_shield_puts(fake_ukc)) == 1

    await env.set_network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
    await env.set_network_policy(_allowlist("pypi.org"))
    assert _shield_puts(fake_ukc)[1:] == [
        [],
        [
            {
                "id": "harbor-egress",
                "direction": "outbound",
                "handler": "passthrough",
                "match": {"hosts": ["pypi.org"], "ports": [80, 443]},
                "priority": 100,
            }
        ],
    ]
    assert env.network_policy == _allowlist("pypi.org")


async def test_policy_change_without_a_shield_fails_clearly(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir)
    await env.start(force_build=False)
    with pytest.raises(RuntimeError, match="without a network shield"):
        await env.set_network_policy(_allowlist("example.com"))


async def test_shield_ports_and_handler_are_configurable(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(
        temp_dir,
        network_policy=_allowlist("github.com"),
        shield_ports=[22, 443],
        shield_handler="http",
        shield_image="acme/shield:2",
        shield_api_image="acme/shield-api:2",
        shield_memory_mb=2048,
    )
    await env.start(force_build=False)
    shield = fake_ukc.created[0]
    assert shield["image"] == "acme/shield:2"
    assert shield["plugins"][0]["image"] == "acme/shield-api:2"
    assert shield["memory_mb"] == 2048
    (policy,) = _shield_puts(fake_ukc)[0]
    assert policy["handler"] == "http"
    assert policy["match"]["ports"] == [22, 443]


async def test_shield_api_waits_for_the_status_endpoint(fake_ukc: FakeClient) -> None:
    fake_ukc.shield_failures = 2
    api = _ShieldApi(
        ApiClientConfig(
            base_url="https://api.test.unikraft.cloud/v1/instances/u/plugins/netshield",
            token="tok",
            http=fake_ukc.shield_http,
        )
    )
    await api.wait_ready(timeout=5)
    statuses = [path for _, path, _ in fake_ukc.shield_requests]
    assert statuses == ["/status", "/status", "/status"]

    request = fake_ukc.shield_http._transport  # the recorder sees the headers below
    assert request is not None
    fake_ukc.shield_failures = 10**6
    with pytest.raises(RuntimeError, match="did not answer within"):
        await api.wait_ready(timeout=0.01)
    await fake_ukc.shield_http.aclose()


async def test_shield_api_reports_rejected_policies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer tok"
        return httpx.Response(422, json={"detail": "Ports [53] are reserved"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        api = _ShieldApi(
            ApiClientConfig(
                base_url="https://x/plugins/netshield", token="tok", http=http
            )
        )
        with pytest.raises(RuntimeError, match="422.*reserved"):
            await api.replace_policies([])
    with pytest.raises(ValueError, match="httpx client"):
        _ShieldApi(ApiClientConfig(base_url="https://x"))


# ---------- exec ----------


async def _started(
    fake_ukc: FakeClient, temp_dir: Path, **kwargs: Any
) -> tuple[UnikraftEnvironment, FakeSandbox]:
    env = _make_env(temp_dir, **kwargs)
    await env.start(force_build=False)
    return env, fake_ukc.sandbox


async def test_exec_before_start_fails(fake_ukc: FakeClient, temp_dir: Path) -> None:
    with pytest.raises(RuntimeError, match="start the environment"):
        await _make_env(temp_dir).exec("true")


async def test_exec_layers_the_environment_like_docker(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(
        fake_ukc,
        temp_dir,
        dockerfile_contents=(
            "FROM python:3.12\n"
            "ENV PATH=/opt/venv/bin:$PATH APP_HOME=/srv\n"
            'ENV GREETING "hello world"\n'
        ),
        config=EnvironmentConfig(env={"TASK": "1"}),
    )
    await env.exec("true", env={"CALL": "2", "TASK": "override"})
    assert sandbox.execs[-1]["env"] == {
        "PATH": "/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/root",
        "APP_HOME": "/srv",
        "GREETING": '"hello world"',
        "TASK": "override",
        "CALL": "2",
    }


async def test_exec_cwd_precedence(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(
        fake_ukc, temp_dir, dockerfile_contents="FROM ubuntu:24.04\nWORKDIR /app\n"
    )
    await env.exec("true")
    assert sandbox.execs[-1]["cwd"] == "/app"

    env.task_env_config.workdir = "/workdir"
    await env.exec("true")
    assert sandbox.execs[-1]["cwd"] == "/workdir"

    await env.exec("true", cwd="/explicit")
    assert sandbox.execs[-1]["cwd"] == "/explicit"


async def test_exec_runs_as_the_requested_user(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(fake_ukc, temp_dir)

    await env.exec("echo hi")
    assert sandbox.execs[-1]["cmd"] == f"{SELECT_SHELL}; exec \"$_shell\" -c 'echo hi'"

    await env.exec("echo hi", user="agent")
    assert sandbox.execs[-1]["cmd"] == (
        f"{SELECT_SHELL}; exec su agent -s \"$_shell\" -c 'echo hi'"
    )

    await env.exec("echo hi", user=1000)
    assert sandbox.execs[-1]["cmd"] == (
        f'{SELECT_SHELL}; exec su "$(getent passwd 1000 | cut -d: -f1)" '
        "-s \"$_shell\" -c 'echo hi'"
    )

    with env.with_default_user("dev"):
        await env.exec("echo hi")
    assert sandbox.execs[-1]["cmd"] == (
        f"{SELECT_SHELL}; exec su dev -s \"$_shell\" -c 'echo hi'"
    )

    await env.exec("echo hi", user="root")
    assert sandbox.execs[-1]["cmd"] == f"{SELECT_SHELL}; exec \"$_shell\" -c 'echo hi'"


async def test_dockerfile_user_is_the_default_user(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(
        fake_ukc, temp_dir, dockerfile_contents="FROM ubuntu:24.04\nUSER agent:agent\n"
    )
    await env.exec("id")
    assert sandbox.execs[-1]["cmd"] == (
        f'{SELECT_SHELL}; exec su agent -s "$_shell" -c id'
    )

    await env.exec("id", user="root")
    assert sandbox.execs[-1]["cmd"] == f'{SELECT_SHELL}; exec "$_shell" -c id'


async def test_exec_reports_a_nonzero_exit_without_raising(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(fake_ukc, temp_dir)
    sandbox.script = lambda cmd: (3, b"partial \xff output", b"boom")
    result = await env.exec("false", timeout_sec=30)
    assert result.return_code == 3
    assert result.stdout == "partial � output"
    assert result.stderr == "boom"
    assert sandbox.execs[-1]["timeout"] == 30
    assert sandbox.execs[-1]["wait_delay"] == 5.0
    assert sandbox.commands["cmd-1"].deleted is True


async def test_exec_timeout_raises_like_docker(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(fake_ukc, temp_dir)

    sandbox.interrupted = True
    with pytest.raises(RuntimeError, match="Command timed out after 5 seconds"):
        await env.exec("sleep 60", timeout_sec=5)

    sandbox.interrupted = False
    sandbox.raise_timeout = True
    with pytest.raises(RuntimeError, match="Command timed out after 5 seconds"):
        await env.exec("sleep 60", timeout_sec=5)

    sandbox.raise_timeout = False
    await env.exec("true")
    assert sandbox.execs[-1]["timeout"] is None
    assert sandbox.execs[-1]["wait_delay"] is None


async def test_exec_streams_output_to_the_callback(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(fake_ukc, temp_dir)
    sandbox.script = lambda cmd: (0, b"", b"")
    sandbox.stream_chunks = [
        OutputChunk("stdout", b"hel"),
        OutputChunk("stderr", b"warn\n"),
        OutputChunk("stdout", "lo é".encode()[:-1]),
        OutputChunk("stdout", "é\n".encode()[1:]),
    ]
    seen: list[tuple[str, str]] = []

    async def callback(text: str, stream: str) -> None:
        seen.append((text, stream))

    with env.scoped_output_callback(callback):
        result = await env.exec("make", user="agent")

    assert seen == [
        ("hel", "stdout"),
        ("warn\n", "stderr"),
        ("lo ", "stdout"),
        ("é\n", "stdout"),
    ]
    assert result.stdout == "hello é\n"
    assert result.stderr == "warn\n"
    assert result.return_code == 0
    assert sandbox.execs[-1]["cmd"] == (
        f'{SELECT_SHELL}; exec su agent -s "$_shell" -c make'
    )
    assert sandbox.commands["cmd-1"].deleted is True


async def test_streamed_exec_timeout_interrupts_the_command(
    fake_ukc: FakeClient,
    fake_build: list[tuple[str, str]],
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, sandbox = await _started(fake_ukc, temp_dir)

    async def stall(self: FakeCommand) -> AsyncIterator[OutputChunk]:
        import asyncio

        await asyncio.sleep(60)
        yield OutputChunk("stdout", b"")

    monkeypatch.setattr(FakeCommand, "stream", stall)

    async def callback(text: str, stream: str) -> None:
        pass

    with env.scoped_output_callback(callback):
        with pytest.raises(RuntimeError, match="Command timed out after 1 seconds"):
            await env.exec("sleep 60", timeout_sec=1)
    assert sandbox.commands["cmd-1"].signals == [2]


# ---------- files ----------


async def test_upload_file_creates_parents_and_chunks_large_files(
    fake_ukc: FakeClient,
    fake_build: list[tuple[str, str]],
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, sandbox = await _started(fake_ukc, temp_dir)
    monkeypatch.setattr(unikraft_module, "_UPLOAD_CHUNK_BYTES", 4)

    small = temp_dir / "small.txt"
    small.write_bytes(b"abc")
    await env.upload_file(small, "/opt/app/small.txt")
    assert sandbox.fs.calls == [("upload", "/opt/app/small.txt", "small.txt", 3, True)]
    assert sandbox.fs.files["/opt/app/small.txt"] == b"abc"

    sandbox.fs.calls.clear()
    large = temp_dir / "large.bin"
    large.write_bytes(b"0123456789")
    await env.upload_file(str(large), "/data/large.bin")
    assert sandbox.fs.calls == [
        ("upload", "/data/large.bin", "large.bin", 4, True),
        ("write", "/data/large.bin", 4, True),
        ("write", "/data/large.bin", 2, True),
    ]
    assert sandbox.fs.files["/data/large.bin"] == b"0123456789"


def _dockerfile(temp_dir: Path, name: str, contents: str) -> Path:
    """Write a Dockerfile the parser will accept: its own directory, exact name."""
    directory = temp_dir / name
    directory.mkdir()
    path = directory / "Dockerfile"
    path.write_text(contents)
    return path


def test_parse_dockerfile_entrypoint_reads_both_forms(temp_dir: Path) -> None:
    exec_form = _dockerfile(
        temp_dir, "exec", 'FROM alpine:3.20\nENTRYPOINT ["/entrypoint.sh", "--serve"]\n'
    )
    assert unikraft_module.parse_dockerfile_entrypoint(exec_form) == [
        "/entrypoint.sh",
        "--serve",
    ]

    shell_form = _dockerfile(
        temp_dir, "shell", "FROM alpine:3.20\nENTRYPOINT /entrypoint.sh --serve\n"
    )
    assert unikraft_module.parse_dockerfile_entrypoint(shell_form) == [
        "/bin/sh",
        "-c",
        "/entrypoint.sh --serve",
    ]

    none_form = _dockerfile(temp_dir, "none", "FROM alpine:3.20\nWORKDIR /app\n")
    assert unikraft_module.parse_dockerfile_entrypoint(none_form) is None
    assert unikraft_module.parse_dockerfile_entrypoint(temp_dir / "Dockerfile") is None


def test_parse_dockerfile_entrypoint_drops_an_earlier_stage(temp_dir: Path) -> None:
    dockerfile = _dockerfile(
        temp_dir,
        "stages",
        'FROM alpine:3.20 AS build\nENTRYPOINT ["/build.sh"]\n'
        "FROM alpine:3.20\nWORKDIR /app\n",
    )
    assert unikraft_module.parse_dockerfile_entrypoint(dockerfile) is None


async def test_the_instance_runs_the_image_entrypoint_before_the_keepalive(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(
        temp_dir,
        dockerfile_contents='FROM alpine:3.20\nENTRYPOINT ["/entrypoint.sh"]\n',
    )

    await env.start(force_build=False)
    assert fake_ukc.created[0]["args"] == [
        "/entrypoint.sh",
        *unikraft_module.DEFAULT_KEEPALIVE,
    ]


async def test_the_instance_runs_the_keepalive_alone_without_an_entrypoint(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir, dockerfile_contents="FROM alpine:3.20\n")

    await env.start(force_build=False)
    assert fake_ukc.created[0]["args"] == list(unikraft_module.DEFAULT_KEEPALIVE)


async def test_pause_suspends_the_instance_and_resume_starts_it(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir)
    await env.start(force_build=False)
    waits_after_start = len(fake_ukc.sandbox.ready_timeouts)

    await env.pause()
    await env.resume()

    uuid = env.get_sandbox_id()
    assert ("suspend", uuid) in fake_ukc.events
    assert ("start", uuid) in fake_ukc.events
    # Resume waits for the plugin, since a suspended instance answers nothing.
    assert len(fake_ukc.sandbox.ready_timeouts) == waits_after_start + 1


async def test_pause_before_start_says_so(temp_dir: Path, fake_ukc: FakeClient) -> None:
    env = _make_env(temp_dir)
    with pytest.raises(RuntimeError, match="before the environment has started"):
        await env.pause()
    with pytest.raises(RuntimeError, match="before the environment has started"):
        await env.resume()


def test_capabilities_declare_pause(temp_dir: Path) -> None:
    assert _make_env(temp_dir).capabilities.pause is True


async def test_upload_file_lands_under_a_directory_that_does_not_exist_yet(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(fake_ukc, temp_dir)

    source = temp_dir / "config.json"
    source.write_bytes(b"{}")
    await env.upload_file(source, "/root/.config/app/config.json")

    assert sandbox.fs.files == {"/root/.config/app/config.json": b"{}"}


async def test_upload_dir_transfers_a_tar_archive(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(fake_ukc, temp_dir)
    source = temp_dir / "tests"
    (source / "nested").mkdir(parents=True)
    (source / "run.sh").write_text("#!/bin/sh\n")
    (source / "run.sh").chmod(0o755)
    (source / "nested" / "link").symlink_to("../run.sh")

    await env.upload_dir(source, "/tests")

    unpack, remove = [inner_command(call["cmd"]) for call in sandbox.execs]
    (archive,) = sandbox.fs.files
    assert archive.startswith("/tmp/.hb-transfer-") and archive.endswith(".tar.gz")
    assert unpack == f"mkdir -p /tests && tar -xzf {archive} -C /tests"
    assert remove == f"rm -f {archive}"
    with tarfile.open(
        fileobj=io.BytesIO(sandbox.fs.files[archive]), mode="r:gz"
    ) as tar:
        members = {member.name: member for member in tar.getmembers()}
    assert members["./run.sh"].mode & 0o111
    assert members["./nested/link"].issym()
    assert members["./nested/link"].linkname == "../run.sh"


async def test_upload_dir_reports_a_failed_unpack(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(fake_ukc, temp_dir)
    source = temp_dir / "src"
    source.mkdir()
    sandbox.script = lambda cmd: (
        (2, b"", b"gzip: truncated") if "tar -xzf" in cmd else (0, b"", b"")
    )
    with pytest.raises(RuntimeError, match="truncated"):
        await env.upload_dir(source, "/src")
    assert inner_command(sandbox.execs[-1]["cmd"]).startswith("rm -f ")


async def test_download_file_writes_bytes(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(fake_ukc, temp_dir)
    sandbox.fs.files["/logs/reward.txt"] = b"1.0\n"
    target = temp_dir / "out" / "deep" / "reward.txt"
    await env.download_file("/logs/reward.txt", target)
    assert target.read_bytes() == b"1.0\n"


async def test_download_dir_packs_remotely_and_extracts_locally(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(fake_ukc, temp_dir)
    remote = temp_dir / "remote"
    (remote / "sub").mkdir(parents=True)
    (remote / "sub" / "a.txt").write_text("A")
    (remote / "b.sh").write_text("#!/bin/sh\n")
    (remote / "b.sh").chmod(0o755)

    def pack(cmd: str) -> tuple[int, bytes, bytes]:
        cmd = inner_command(cmd)
        if cmd.startswith("tar -czf "):
            archive = cmd.split()[2]
            sandbox.fs.files[archive] = pack_dir_to_bytes(
                remote, compress=True
            ).getvalue()
        return (0, b"", b"")

    sandbox.script = pack
    target = temp_dir / "local"
    await env.download_dir("/logs/agent", target)

    assert (target / "sub" / "a.txt").read_text() == "A"
    assert (target / "b.sh").stat().st_mode & 0o111
    pack_cmd, remove_cmd = [inner_command(call["cmd"]) for call in sandbox.execs]
    assert pack_cmd.endswith("-C /logs/agent .")
    assert remove_cmd.startswith("rm -f /tmp/.hb-transfer-")


async def test_download_dir_reports_a_failed_pack(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env, sandbox = await _started(fake_ukc, temp_dir)
    sandbox.script = lambda cmd: (1, b"", b"tar: /nope: No such file")
    with pytest.raises(RuntimeError, match="No such file"):
        await env.download_dir("/nope", temp_dir / "local")


# ---------- stop ----------


async def test_stop_deletes_instance_and_shield_and_closes_the_client(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir, network_policy=_allowlist("example.com"))
    await env.start(force_build=False)
    await env.stop(delete=True)
    assert fake_ukc.events == [("delete", "inst-2"), ("delete", "inst-1")]
    assert fake_ukc.closed is True
    assert env.get_sandbox_id() is None
    with pytest.raises(RuntimeError, match="start the environment"):
        await env.exec("true")


async def test_stop_without_delete_stops_both_instances(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir, network_policy=_allowlist("example.com"))
    await env.start(force_build=False)
    await env.stop(delete=False)
    assert fake_ukc.events == [("stop", "inst-2"), ("stop", "inst-1")]
    assert fake_ukc.closed is True
    assert env.get_sandbox_id() == "inst-2"


async def test_stop_falls_back_to_stopping_before_deleting(
    fake_ukc: FakeClient, fake_build: list[tuple[str, str]], temp_dir: Path
) -> None:
    env = _make_env(temp_dir)
    await env.start(force_build=False)
    fake_ukc.fail_delete_once = True
    await env.stop(delete=True)
    assert fake_ukc.events == [
        ("delete", "inst-1"),
        ("stop", "inst-1"),
        ("delete", "inst-1"),
    ]


async def test_stop_before_start_is_a_noop(
    fake_ukc: FakeClient, temp_dir: Path
) -> None:
    await _make_env(temp_dir).stop(delete=True)
    assert fake_ukc.events == []
    assert fake_ukc.closed is False


# ---------- preflight and the build runner ----------


def test_preflight_requires_token_and_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UKC_TOKEN", raising=False)
    monkeypatch.delenv("HARBOR_UNIKRAFT_CLI", raising=False)
    with pytest.raises(SystemExit, match="UKC_TOKEN"):
        UnikraftEnvironment.preflight()

    monkeypatch.setenv("UKC_TOKEN", "tok")
    monkeypatch.delenv("UKC_METRO", raising=False)
    with pytest.raises(SystemExit, match="UKC_METRO"):
        UnikraftEnvironment.preflight()

    monkeypatch.setenv("UKC_METRO", "fra0-fe-test")
    monkeypatch.setattr(unikraft_module.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="'unikraft' CLI"):
        UnikraftEnvironment.preflight()

    monkeypatch.setenv("HARBOR_UNIKRAFT_CLI", "/opt/unikraft")
    monkeypatch.setattr(
        unikraft_module.shutil,
        "which",
        lambda name: name if name == "/opt/unikraft" else None,
    )
    UnikraftEnvironment.preflight()


async def test_run_build_invokes_the_cli(
    fake_ukc: FakeClient, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []
    outcomes: list[tuple[int, bytes]] = []

    class FakeProcess:
        def __init__(self, returncode: int, stderr: bytes) -> None:
            self.returncode = returncode
            self._stderr = stderr

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", self._stderr

        def kill(self) -> None:
            pass

        async def wait(self) -> int:
            return self.returncode

    async def create_subprocess_exec(*args: str, **kwargs: Any) -> FakeProcess:
        calls.append(args)
        returncode, stderr = outcomes.pop(0)
        return FakeProcess(returncode, stderr)

    monkeypatch.setattr(
        unikraft_module.asyncio, "create_subprocess_exec", create_subprocess_exec
    )
    monkeypatch.setattr(UnikraftEnvironment._run_build.retry, "sleep", AsyncMock())  # type: ignore[attr-defined]
    monkeypatch.setenv("HARBOR_UNIKRAFT_CLI", "/opt/cli/unikraft")

    env = _make_env(temp_dir)
    dockerfile = env.environment_dir / "Dockerfile"
    outcomes.append((0, b""))
    await env._run_build(dockerfile, "demo/harbor-hello-world:abc")
    assert calls == [
        (
            "/opt/cli/unikraft",
            "build",
            str(dockerfile),
            "--arch",
            "x86_64",
            "-o",
            "demo/harbor-hello-world:abc",
        )
    ]

    outcomes.extend([(1, b"error: no such base image"), (1, b"error: still broken")])
    with pytest.raises(RuntimeError, match=r"(?s)exit 1.*still broken"):
        await env._run_build(dockerfile, "demo/harbor-hello-world:abc")
    assert len(calls) == 3


# ---------- Dockerfile parsing ----------


def test_parse_dockerfile_env_expands_and_resets_per_stage(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM golang:1.22 AS build\n"
        "ENV GOPATH=/go\n"
        "FROM ubuntu:24.04\n"
        "ENV APP=/srv/app PATH=$APP/bin:${PATH}\n"
        "ENV NAME John Doe\n"
        'ENV QUOTED="a b" EMPTY=\n'
        "ENV MISSING=$UNSET/x\n"
    )
    assert parse_dockerfile_env(dockerfile) == {
        "APP": "/srv/app",
        "PATH": "/srv/app/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "NAME": "John Doe",
        "QUOTED": "a b",
        "EMPTY": "",
        "MISSING": "/x",
    }
    assert parse_dockerfile_env(tmp_path / "missing") == {}


def test_parse_dockerfile_user_takes_the_last_stage(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM a\nUSER build\nFROM b\nUSER agent:agent\n")
    assert parse_dockerfile_user(dockerfile) == "agent"
    dockerfile.write_text("FROM a\nUSER build\nFROM b\n")
    assert parse_dockerfile_user(dockerfile) is None
    assert parse_dockerfile_user(tmp_path / "missing") is None


def test_sanitize_name_gives_a_valid_platform_name() -> None:
    # The platform rejects a name with two hyphens in sequence, a trailing
    # hyphen, or 64 characters or more.
    assert _sanitize_name("task-demo-_vohe2__BRDNdXW") == "task-demo-vohe2-brdndxw"
    assert _sanitize_name("Task__Demo") == "task-demo"
    assert _sanitize_name("--task--demo--") == "task-demo"
    assert _sanitize_name("_" * 5) == "harbor"
    long_name = _sanitize_name("task-" + "a" * 80)
    assert len(long_name) == 40 and not long_name.endswith("-")


def test_run_as_user_quotes_the_command() -> None:
    assert _run_as_user("echo 'it works'", "agent") == (
        f"{SELECT_SHELL}; exec su agent -s \"$_shell\" -c 'echo '\"'\"'it works'\"'\"''"
    )
    assert _run_as_user("id", "1000") == (
        f'{SELECT_SHELL}; exec su "$(getent passwd 1000 | cut -d: -f1)" '
        '-s "$_shell" -c id'
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="the host has no bash")
def test_run_in_shell_runs_bash_syntax() -> None:
    # Harbor prefixes agent commands with `set -o pipefail`, which a POSIX
    # shell such as dash rejects.
    result = subprocess.run(
        ["/bin/sh", "-c", _run_in_shell("set -o pipefail; echo ok")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_run_in_shell_falls_back_to_sh(tmp_path: Path) -> None:
    result = subprocess.run(
        ["/bin/sh", "-c", _run_in_shell("echo ok")],
        capture_output=True,
        text=True,
        env={"PATH": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_image_repository_drops_scheme_registry_digest_and_tag() -> None:
    # The store reports every image under the central registry, so the
    # repository and tag are what identify one, whichever registry holds it.
    assert (
        _image_repository("unikraft.io/demo/harbor-hello-world@sha256:abc")
        == "demo/harbor-hello-world"
    )
    assert _image_repository("oci://unikraft.io/demo/app:latest") == "demo/app"
    assert _image_repository("index.fra0-fe-test.unikraft.cloud/demo/app") == "demo/app"
    assert _image_repository("demo/app:tag") == "demo/app"
    assert _image_repository("") == ""


async def test_create_waits_out_an_image_the_nodes_cannot_pull_yet(
    fake_ukc: FakeClient,
    fake_build: list[tuple[str, str]],
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(unikraft_module, "_IMAGE_RETRY_INTERVAL_SEC", 0)
    env = _make_env(temp_dir)
    fake_ukc.create_failures = 2

    await env.start(force_build=False)

    assert len(fake_ukc.created) == 3
    # Each failed attempt is discarded, so no stopped instance is left behind.
    names = {name for verb, name in fake_ukc.events if verb == "delete"}
    assert names == {f"name:{fake_ukc.created[0]['name']}"}
    assert env.get_sandbox_id() == "inst-3"


async def test_create_is_retried_for_an_image_this_run_did_not_push(
    fake_ukc: FakeClient,
    fake_build: list[tuple[str, str]],
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node fails to pull whatever it is asked for, not only a new image."""
    monkeypatch.setattr(unikraft_module, "_IMAGE_RETRY_INTERVAL_SEC", 0)
    env = _make_env(temp_dir)
    fake_ukc.registry.add(env._image_ref())
    fake_ukc.create_failures = 1

    await env.start(force_build=False)
    assert fake_build == []
    assert len(fake_ukc.created) == 2


async def test_the_shield_create_is_retried_like_any_other(
    fake_ukc: FakeClient,
    fake_build: list[tuple[str, str]],
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(unikraft_module, "_IMAGE_RETRY_INTERVAL_SEC", 0)
    env = _make_env(
        temp_dir, network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    )
    fake_ukc.create_failures = 1

    await env.start(force_build=False)
    # The shield is attempted twice, then the task instance follows it.
    names = [spec["name"] for spec in fake_ukc.created]
    assert names == [env._shield_name, env._shield_name, env._instance_name]


async def test_a_failed_create_reports_why_the_platform_stopped_it(
    fake_ukc: FakeClient,
    fake_build: list[tuple[str, str]],
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(unikraft_module, "_IMAGE_RETRY_INTERVAL_SEC", 0)
    monkeypatch.setattr(unikraft_module, "_IMAGE_RETRY_ATTEMPTS", 2)
    env = _make_env(temp_dir)
    fake_ukc.create_failures = 99
    fake_ukc.stop_detail = (4, 1)

    with pytest.raises(UnikraftCloudError, match="could not pull the image"):
        await env.start(force_build=False)


def test_kernel_stop_codes_are_decoded() -> None:
    # 0xC0004: page fault with errno 12, which is how a too-small memory_mb reads.
    assert "ran out of memory" in unikraft_module._kernel_stop_detail(0xC0004)
    assert "segmentation fault" in unikraft_module._kernel_stop_detail(5)
    assert "code 255" in unikraft_module._kernel_stop_detail(255)


async def test_a_failed_create_reports_an_out_of_memory_kernel_stop(
    fake_ukc: FakeClient,
    fake_build: list[tuple[str, str]],
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(unikraft_module, "_IMAGE_RETRY_INTERVAL_SEC", 0)
    monkeypatch.setattr(unikraft_module, "_IMAGE_RETRY_ATTEMPTS", 2)
    env = _make_env(temp_dir)
    fake_ukc.create_failures = 99
    fake_ukc.stop_detail = (1, 0xC0004)

    with pytest.raises(UnikraftCloudError, match="ran out of memory"):
        await env.start(force_build=False)


async def test_create_gives_up_after_the_last_attempt(
    fake_ukc: FakeClient,
    fake_build: list[tuple[str, str]],
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(unikraft_module, "_IMAGE_RETRY_INTERVAL_SEC", 0)
    monkeypatch.setattr(unikraft_module, "_IMAGE_RETRY_ATTEMPTS", 3)
    env = _make_env(temp_dir)
    fake_ukc.create_failures = 99

    with pytest.raises(UnikraftCloudError):
        await env.start(force_build=False)
    assert len(fake_ukc.created) == 3
