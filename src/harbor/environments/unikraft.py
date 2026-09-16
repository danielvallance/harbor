from __future__ import annotations

import asyncio
import base64
import codecs
import contextlib
import json
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, override
from uuid import uuid4

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.definition import (
    COMPOSE_FILE_NAME,
    DOCKERFILE_NAME,
    SNAPSHOT_HASH_LEN,
    effective_exec_cwd,
    parse_dockerfile_workdir,
    require_agent_environment_definition,
    should_use_prebuilt_docker_image,
)
from harbor.environments.lifecycle_timing import timed_environment_subphase
from harbor.environments.tar_transfer import (
    extract_dir_from_bytes,
    pack_dir_to_bytes,
    remote_pack_command,
    remote_unpack_command,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import TrialPaths
from harbor.utils.optional_import import MissingExtraError

try:
    from unikraft_cloud import UnikraftCloud, UnikraftCloudError
    from unikraft_cloud.core.http import ApiClientConfig
    from unikraft_cloud.plugins.sandbox import ExecTimeoutError, OutputChunk, Sandbox

    _HAS_UNIKRAFT = True
except ImportError:
    _HAS_UNIKRAFT = False


#: The name the sandbox plugin is attached under, and hence its route segment.
PLUGIN_NAME = "sandbox"
#: The name the network shield's control plugin is attached under.
SHIELD_PLUGIN_NAME = "netshield"
DEFAULT_PLUGIN_IMAGE = "plugins/sandbox:staging"
DEFAULT_SHIELD_IMAGE = "demo/netshield:latest"
DEFAULT_SHIELD_API_IMAGE = "demo/netshield-api:latest"
#: Every other Harbor backend allows all ports; the shield needs them listed.
DEFAULT_SHIELD_PORTS: tuple[int, ...] = (80, 443)
DEFAULT_SHIELD_HANDLER = "passthrough"
DEFAULT_SHIELD_MEMORY_MB = 1024
#: The platform default of 128 MB cannot run an agent.
DEFAULT_MEMORY_MB = 2048
DEFAULT_VCPUS = 1
#: The guest architecture must match the metro's hosts.
DEFAULT_ARCH = "x86_64"
#: Keeps the instance alive; the plugin runs every command beside it.
DEFAULT_KEEPALIVE: tuple[str, ...] = ("/bin/sh", "-c", "sleep infinity")
UNIKRAFT_CLI_ENV = "HARBOR_UNIKRAFT_CLI"

#: Base64 in one request must stay under the plugin's 2 MB body limit, so
#: large files travel in appended pieces.
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_TRANSFER_DIR = PurePosixPath("/tmp")
#: How long an interrupted command may take to exit before it is given up on.
_INTERRUPT_GRACE_SEC = 5.0
_SHIELD_REQUEST_TIMEOUT_SEC = 30.0
_READY_FIRST_INTERVAL_SEC = 0.25
_READY_MAX_INTERVAL_SEC = 2.0
_BUILD_LOG_TAIL_BYTES = 16 * 1024
#: How often to retry a create whose image the nodes cannot pull yet.
_IMAGE_RETRY_INTERVAL_SEC = 5.0
_IMAGE_RETRY_ATTEMPTS = 6
#: The registry that holds task images when no other one is pinned.
DEFAULT_REGISTRY = "unikraft.io"
#: The manifest types a registry may answer an existence check with.
_MANIFEST_TYPES = (
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)
_REGISTRY_TIMEOUT_SEC = 30.0
#: The stop of an instance that the platform alone brought down.
_STOP_REASON_PLATFORM = 4
#: The stop of an instance whose kernel exited.
_STOP_REASON_KERNEL = 1
#: The platform stop code for an image a node could not pull.
_STOP_CODE_IMAGE_PULL_FAILED = 1
#: A kernel stop code packs the reason in its low byte and an errno above it.
_KERNEL_STOP_REASONS = {
    1: "an assertion failed",
    2: "an arithmetic error",
    3: "an instruction error",
    4: "a page fault",
    5: "a segmentation fault",
    6: "a hardware error",
    7: "a security violation",
}
_ENOMEM = 12
#: The longest the create call may block for the instance to run.
_CREATE_WAIT_MAX_SEC = 60
#: Instance and interface names must stay DNS-label sized.
_NAME_BASE_LEN = 40
_DOCKER_DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _sanitize_name(name: str) -> str:
    """Return a lower-case DNS-label prefix for an instance name."""
    name = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return name[:_NAME_BASE_LEN].rstrip("-") or "harbor"


def _sanitize_image_name(name: str) -> str:
    """Return a registry path segment for an image name."""
    name = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-.")
    return name or "harbor"


def _image_repository(url: str) -> str:
    """The ``<namespace>/<name>`` of an image URL, without scheme, registry or digest.

    The image store reports every image under the central registry, whichever
    one holds it, so the repository and tag are what identify one there.
    """
    reference = url.split("://", 1)[-1].split("@", 1)[0]
    head, _, name = reference.rpartition("/")
    reference = f"{head}/{name.split(':', 1)[0]}" if head else name.split(":", 1)[0]
    # A registry host carries a dot or a port, unlike a namespace.
    host, _, rest = reference.partition("/")
    return rest if rest and ("." in host or ":" in host) else reference


def _auth_challenge(header: str) -> dict[str, str]:
    """The parameters of a registry's ``Bearer`` authentication challenge."""
    scheme, _, rest = header.partition(" ")
    if scheme.lower() != "bearer":
        return {}
    return dict(re.findall(r'([a-z_]+)="([^"]*)"', rest))


def _registry_credentials(token: str | None) -> tuple[str, str] | None:
    """The user and password the registry wants, which a UKC token encodes."""
    if not token:
        return None
    try:
        decoded = base64.b64decode(token, validate=True).decode()
    except (ValueError, UnicodeDecodeError):
        return None
    user, separator, password = decoded.partition(":")
    return (user, password) if separator and user else None


def _kernel_stop_detail(stop_code: int) -> str:
    """Why a kernel stopped, from the reason and errno its stop code packs."""
    reason = stop_code & 0xFF
    errno = (stop_code >> 16) & 0xFF
    if reason == 4 and errno == _ENOMEM:
        return "the instance ran out of memory; its image may not fit in memory_mb"
    described = _KERNEL_STOP_REASONS.get(reason)
    if described is None:
        return f"the kernel exited, code {stop_code}"
    return f"the kernel exited on {described}" + (f" (errno {errno})" if errno else "")


def _run_as_user(command: str, user: str | int) -> str:
    """Wrap ``command`` so ``su`` runs it as ``user`` from the same directory."""
    if isinstance(user, int) or str(user).isdigit():
        user_arg = f'"$(getent passwd {int(user)} | cut -d: -f1)"'
    else:
        user_arg = shlex.quote(str(user))
    return f"su {user_arg} -s /bin/sh -c {shlex.quote(command)}"


def _is_root(user: str | int | None) -> bool:
    return user is None or str(user) in ("root", "0")


def _expand(value: str, env: Mapping[str, str]) -> str:
    """Substitute ``$VAR`` and ``${VAR}`` the way the Docker builder does."""
    return _VARIABLE.sub(
        lambda match: env.get(match.group(1) or match.group(2), ""), value
    )


def _split_env(value: str) -> list[tuple[str, str]]:
    """Split one ENV instruction into pairs, in either of its two forms."""
    try:
        tokens = shlex.split(value)
    except ValueError:
        tokens = value.split()
    if tokens and "=" in tokens[0]:
        pairs: list[tuple[str, str]] = []
        for token in tokens:
            key, separator, item = token.partition("=")
            if separator and key:
                pairs.append((key, item))
        return pairs
    key, _, rest = value.partition(" ")
    return [(key, rest.strip())] if key else []


def parse_dockerfile_env(dockerfile_path: Path) -> dict[str, str]:
    """Return the final build stage's ENV, expanded as the builder would.

    Each FROM starts a new stage, so the variables of an earlier stage are
    dropped when one begins.
    """
    if not dockerfile_path.exists():
        return {}
    from dockerfile_parse import DockerfileParser

    env: dict[str, str] = {}
    for instruction in DockerfileParser(path=str(dockerfile_path)).structure:
        name = instruction.get("instruction")
        if name == "FROM":
            env = {}
            continue
        if name != "ENV":
            continue
        known = {"PATH": _DOCKER_DEFAULT_PATH, **env}
        for key, raw in _split_env(str(instruction.get("value", "")).strip()):
            env[key] = _expand(raw, known)
            known[key] = env[key]
    return env


def _command_words(value: str) -> list[str] | None:
    """The words of a Dockerfile command, in either the exec or the shell form."""
    if not value:
        return None
    if value.startswith("["):
        try:
            words = json.loads(value)
        except ValueError:
            return None
        return [str(word) for word in words] if isinstance(words, list) else None
    # The shell form runs through a shell, as the Docker builder does.
    return ["/bin/sh", "-c", value]


def parse_dockerfile_entrypoint(dockerfile_path: Path) -> list[str] | None:
    """Return the final build stage's ENTRYPOINT, or None.

    Each FROM starts a new stage, so an earlier stage's entrypoint is dropped
    when one begins.
    """
    if not dockerfile_path.exists():
        return None
    from dockerfile_parse import DockerfileParser

    entrypoint: list[str] | None = None
    for instruction in DockerfileParser(path=str(dockerfile_path)).structure:
        name = instruction.get("instruction")
        if name == "FROM":
            entrypoint = None
        elif name == "ENTRYPOINT":
            entrypoint = _command_words(str(instruction.get("value", "")).strip())
    return entrypoint


def parse_dockerfile_user(dockerfile_path: Path) -> str | None:
    """Return the final build stage's USER, without its group, or None."""
    if not dockerfile_path.exists():
        return None
    from dockerfile_parse import DockerfileParser

    user: str | None = None
    for instruction in DockerfileParser(path=str(dockerfile_path)).structure:
        name = instruction.get("instruction")
        if name == "FROM":
            user = None
        elif name == "USER":
            value = str(instruction.get("value", "")).strip()
            user = value.split(":", 1)[0] or None
    return user


class _ShieldApi:
    """The network shield's control API, reached through its plugin route."""

    def __init__(self, config: ApiClientConfig) -> None:
        if config.http is None:
            raise ValueError("The shield client needs the session's httpx client.")
        self._http = config.http
        self._base_url = f"{config.base_url.rstrip('/')}/api/v1"
        self._headers = {"accept": "application/json"}
        if config.token:
            self._headers["authorization"] = f"Bearer {config.token}"

    async def _request(
        self, method: str, path: str, *, json: Any | None = None
    ) -> httpx.Response:
        response = await self._http.request(
            method,
            f"{self._base_url}{path}",
            json=json,
            headers=self._headers,
            timeout=_SHIELD_REQUEST_TIMEOUT_SEC,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Network shield {method} {path} failed with "
                f"{response.status_code}: {response.text[:500]}"
            )
        return response

    async def status(self) -> dict[str, Any]:
        response = await self._request("GET", "/status")
        data: dict[str, Any] = response.json()
        return data

    async def wait_ready(self, timeout: float) -> None:
        """Poll the status endpoint until it answers, backing off in between."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        interval = _READY_FIRST_INTERVAL_SEC
        while True:
            try:
                await self.status()
                return
            except (httpx.HTTPError, RuntimeError) as exc:
                if loop.time() >= deadline:
                    raise RuntimeError(
                        f"Network shield did not answer within {timeout}s: {exc}"
                    ) from exc
            await asyncio.sleep(interval)
            interval = min(interval * 2, _READY_MAX_INTERVAL_SEC)

    async def replace_policies(self, policies: list[dict[str, Any]]) -> None:
        """Install ``policies`` as the shield's whole policy set."""
        await self._request("PUT", "/policies", json=policies)


class UnikraftEnvironment(BaseEnvironment):
    """A Harbor environment backed by a Unikraft Cloud microVM.

    The task image is built from its Dockerfile with the ``unikraft`` CLI and
    started with the sandbox plugin attached, which runs commands and moves
    files over the instance's authenticated plugin route. A task whose network
    policy is not public gets its own network shield, a relay microVM that the
    instance's traffic passes through.
    """

    provider_name = "unikraft"

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *,
        metro: str | None = None,
        image_namespace: str | None = None,
        plugin_image: str = DEFAULT_PLUGIN_IMAGE,
        default_memory_mb: int = DEFAULT_MEMORY_MB,
        default_vcpus: int = DEFAULT_VCPUS,
        keepalive: Sequence[str] = DEFAULT_KEEPALIVE,
        ready_timeout_sec: float = 120.0,
        autokill_after_stop_sec: int = 3600,
        unikraft_cli: str | None = None,
        arch: str = DEFAULT_ARCH,
        registry: str | None = None,
        require_prebuilt_image: bool = False,
        shield_image: str = DEFAULT_SHIELD_IMAGE,
        shield_api_image: str = DEFAULT_SHIELD_API_IMAGE,
        shield_ports: Sequence[int] = DEFAULT_SHIELD_PORTS,
        shield_handler: str = DEFAULT_SHIELD_HANDLER,
        shield_memory_mb: int = DEFAULT_SHIELD_MEMORY_MB,
        **kwargs: Any,
    ) -> None:
        if not _HAS_UNIKRAFT:
            raise MissingExtraError(package="unikraft-cloud", extra="unikraft")
        if default_memory_mb < 1 or default_vcpus < 1:
            raise ValueError(
                "Unikraft default_memory_mb and default_vcpus must be positive."
            )
        if not shield_ports:
            raise ValueError("Unikraft shield_ports must name at least one port.")
        if shield_handler not in ("passthrough", "http"):
            raise ValueError("Unikraft shield_handler must be 'passthrough' or 'http'.")

        self._metro = metro or os.environ.get("UKC_METRO")
        self._image_namespace = image_namespace
        self._plugin_image = plugin_image
        self._default_memory_mb = default_memory_mb
        self._default_vcpus = default_vcpus
        self._keepalive = list(keepalive)
        self._ready_timeout_sec = ready_timeout_sec
        self._autokill_after_stop_sec = autokill_after_stop_sec
        self._unikraft_cli = (
            unikraft_cli or os.environ.get(UNIKRAFT_CLI_ENV) or "unikraft"
        )
        self._arch = arch
        self._registry = registry
        self._require_prebuilt_image = require_prebuilt_image
        self._shield_image = shield_image
        self._shield_api_image = shield_api_image
        self._shield_ports = [int(port) for port in shield_ports]
        self._shield_handler = shield_handler
        self._shield_memory_mb = shield_memory_mb

        self._ukc: UnikraftCloud | None = None
        self._sandbox: Sandbox | None = None
        self._instance_uuid: str | None = None
        self._shield_uuid: str | None = None
        self._shield_interface_uuid: str | None = None
        self._shield_api: _ShieldApi | None = None
        self._name_base = f"{_sanitize_name(session_id)}-{uuid4().hex[:7]}"

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        dockerfile = self.environment_dir / DOCKERFILE_NAME
        self._dockerfile_workdir = parse_dockerfile_workdir(dockerfile)
        self._dockerfile_env = parse_dockerfile_env(dockerfile)
        self._dockerfile_user = parse_dockerfile_user(dockerfile)
        self._dockerfile_entrypoint = parse_dockerfile_entrypoint(dockerfile)

    @staticmethod
    @override
    def type() -> EnvironmentType:
        return EnvironmentType.UNIKRAFT

    @classmethod
    @override
    def preflight(cls) -> None:
        if not _HAS_UNIKRAFT:
            raise SystemExit(
                "Unikraft requires the 'unikraft-cloud' package. Install it with "
                "'pip install harbor[unikraft]' or 'uv tool install harbor[unikraft]'."
            )
        if not os.environ.get("UKC_TOKEN"):
            raise SystemExit(
                "Unikraft requires UKC_TOKEN to be set. "
                "Please set this environment variable and try again."
            )
        if not os.environ.get("UKC_METRO"):
            raise SystemExit(
                "Unikraft requires UKC_METRO to name the metro that runs the "
                "instances, as a code or an API base URL."
            )
        cli = os.environ.get(UNIKRAFT_CLI_ENV) or "unikraft"
        if shutil.which(cli) is None:
            raise SystemExit(
                f"Unikraft requires the '{cli}' CLI to build task images. Install it, "
                f"or point {UNIKRAFT_CLI_ENV} at the binary."
            )

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        # vCPUs and memory size a microVM, so both are hard ceilings.
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        # The shield filters by host pattern and refuses unknown names at DNS;
        # its policies are replaced at run time, so phases can switch policy.
        return EnvironmentCapabilities(
            disable_internet=True,
            network_allowlist=True,
            network_allowlist_hostnames=True,
            network_allowlist_wildcard_hostnames=True,
            dynamic_network_policy=True,
            pause=True,
        )

    @override
    def _validate_definition(self) -> None:
        if (self.environment_dir / COMPOSE_FILE_NAME).exists():
            raise ValueError(
                f"{self.type().value} environment does not support Docker Compose "
                "task environments."
            )
        require_agent_environment_definition(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
        )

    @override
    def get_sandbox_id(self) -> str | None:
        return self._instance_uuid

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _client(self) -> UnikraftCloud:
        if self._ukc is None:
            # Creating an instance targets one metro, so the client is pinned.
            if not self._metro:
                raise ValueError(
                    "Unikraft needs a metro to run the instance in: pass the "
                    "metro kwarg or set UKC_METRO."
                )
            self._ukc = UnikraftCloud(metro=self._metro)
        return self._ukc

    def _require_sandbox(self) -> Sandbox:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        return self._sandbox

    def _needs_shield(self) -> bool:
        policies = [self.network_policy, *self._phase_network_policies]
        return any(policy.network_mode != NetworkMode.PUBLIC for policy in policies)

    @property
    def _instance_name(self) -> str:
        return self._name_base

    @property
    def _shield_name(self) -> str:
        return f"{self._name_base}-shield"

    def _image_ref(self) -> str:
        namespace = self._image_namespace or os.environ.get("UKC_USER")
        if not namespace:
            raise ValueError(
                "Unikraft needs a registry namespace for task images: pass the "
                "image_namespace kwarg or set UKC_USER."
            )
        name = _sanitize_image_name(f"harbor-{self.environment_name}")
        tag = self.environment_id[:SNAPSHOT_HASH_LEN]
        # Unqualified, so the CLI resolves it against the metro's own index and
        # falls back to the central registry. `registry` pins one instead.
        prefix = f"{self._registry}/" if self._registry else ""
        return f"{prefix}{namespace}/{name}:{tag}"

    def _tags(self) -> list[str]:
        tags = ["harbor", f"harbor.session={self.session_id}"]
        if self.context_id is not None:
            tags.append(f"harbor.context={self.context_id}")
        return tags

    def _autokill(self) -> dict[str, int]:
        return {"time_ms": self._autokill_after_stop_sec * 1000}

    def _create_wait_sec(self) -> int:
        """How long the create blocks for a running instance; the plugin wait covers the rest."""
        return max(0, min(int(self._ready_timeout_sec), _CREATE_WAIT_MAX_SEC))

    def _instance_args(self) -> list[str]:
        """The instance command line: the image's entrypoint, then the keepalive.

        The platform keeps an image's CMD but drops its ENTRYPOINT, so the
        entrypoint is supplied here. Docker keeps it and replaces the command,
        which is what the keepalive stands in for.
        """
        return [*(self._dockerfile_entrypoint or []), *self._keepalive]

    def _instance_spec(self, image_ref: str) -> dict[str, Any]:
        memory_mb = self._resource_limit_value("memory", auto_mode=ResourceMode.LIMIT)
        vcpus = self._resource_limit_value("cpu", auto_mode=ResourceMode.LIMIT)
        spec: dict[str, Any] = {
            "name": self._instance_name,
            "image": image_ref,
            "memory_mb": memory_mb or self._default_memory_mb,
            "vcpus": vcpus or self._default_vcpus,
            "args": self._instance_args(),
            "plugins": [
                {"name": PLUGIN_NAME, "image": self._plugin_image, "config": {}}
            ],
            "tags": self._tags(),
            "restart_policy": "never",
            "autokill": self._autokill(),
            "autostart": True,
            "timeout_s": self._create_wait_sec(),
        }
        startup_env = self._startup_env()
        if startup_env:
            spec["env"] = startup_env
        if self._shield_uuid is not None:
            # A relayed instance serves no public traffic, so it has no service.
            spec["network_interfaces"] = [
                {"relay": {"uuid": self._shield_interface_uuid, "relay_dns": True}}
            ]
        return spec

    def _shield_spec(self) -> dict[str, Any]:
        return {
            "name": self._shield_name,
            "image": self._shield_image,
            "memory_mb": self._shield_memory_mb,
            "plugins": [
                {
                    "name": SHIELD_PLUGIN_NAME,
                    "image": self._shield_api_image,
                    "config": {},
                }
            ],
            "tags": self._tags(),
            "restart_policy": "never",
            "autokill": self._autokill(),
            "autostart": True,
            "timeout_s": self._create_wait_sec(),
        }

    def _shield_policies(self, network_policy: NetworkPolicy) -> list[dict[str, Any]]:
        """The shield policy set that enforces ``network_policy``.

        The shield denies whatever no policy allows, so no-network is an empty
        set and public through a shield is every host on the allowed ports.
        """
        if network_policy.network_mode == NetworkMode.NO_NETWORK:
            return []
        if network_policy.network_mode == NetworkMode.PUBLIC:
            hosts = ["*"]
        else:
            hosts = list(network_policy.allowed_hosts)
        return [
            {
                "id": "harbor-egress",
                "direction": "outbound",
                "handler": self._shield_handler,
                "match": {"hosts": hosts, "ports": list(self._shield_ports)},
                "priority": 100,
            }
        ]

    async def _registry_token(
        self, client: httpx.AsyncClient, challenge: dict[str, str], repository: str
    ) -> str | None:
        """Trade the UKC credentials for a registry token, as the challenge asks."""
        realm = challenge.get("realm")
        credentials = _registry_credentials(os.environ.get("UKC_TOKEN"))
        if not realm or credentials is None:
            return None
        params = {"scope": f"repository:{repository}:pull"}
        if challenge.get("service"):
            params["service"] = challenge["service"]
        response = await client.get(realm, params=params, auth=credentials)
        response.raise_for_status()
        body = response.json()
        token = body.get("token") or body.get("access_token")
        return token if isinstance(token, str) else None

    async def _image_exists(self, image_ref: str) -> bool:
        """Whether the registry holds this exact image.

        The tag is the hash of the environment's contents, so a match is the
        same build inputs. The registry answers what a node can pull. The image
        store does not: it reports the nodes' cache, which drops images the
        registry keeps and keeps images the registry has dropped.
        """
        repository = _image_repository(image_ref)
        tag = image_ref.rpartition(":")[2]
        registry = self._registry or DEFAULT_REGISTRY
        url = f"https://{registry}/v2/{repository}/manifests/{tag}"
        headers: dict[str, str] = {"accept": ", ".join(_MANIFEST_TYPES)}
        try:
            async with httpx.AsyncClient(
                timeout=_REGISTRY_TIMEOUT_SEC, follow_redirects=True
            ) as client:
                response = await client.head(url, headers=headers)
                if response.status_code == httpx.codes.UNAUTHORIZED:
                    challenge = _auth_challenge(
                        response.headers.get("www-authenticate", "")
                    )
                    token = await self._registry_token(client, challenge, repository)
                    if token is None:
                        raise RuntimeError("the registry issued no token")
                    headers["authorization"] = f"Bearer {token}"
                    response = await client.head(url, headers=headers)
        except Exception as exc:
            self.logger.warning(
                f"Could not check for image {image_ref}, building it: {exc}"
            )
            return False
        return response.status_code == httpx.codes.OK

    def _assert_prebuilt(self, image_ref: str, force_build: bool) -> None:
        """Refuse a build request when the caller requires a prebuilt image."""
        if force_build:
            raise ValueError(
                f"Unikraft cannot rebuild {image_ref}: force_build asks for a "
                "build, and require_prebuilt_image refuses one. Build the "
                "image before the run, or unset require_prebuilt_image."
            )

    async def _assert_image_exists(self, image_ref: str) -> None:
        """Fail with an actionable message when the prebuilt image is absent."""
        if not await self._image_exists(image_ref):
            raise RuntimeError(
                f"Unikraft image {image_ref} does not exist. "
                "require_prebuilt_image is set, so the run needs the image to "
                "be built and pushed before it starts."
            )
        self.logger.debug(f"Using prebuilt image {image_ref}")

    async def _build_image(self, image_ref: str, force_build: bool) -> None:
        """Build the task image from its Dockerfile, or wrap a prebuilt one."""
        use_prebuilt = should_use_prebuilt_docker_image(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
            force_build=force_build,
        )
        if not use_prebuilt:
            await self._run_build(self.environment_dir / DOCKERFILE_NAME, image_ref)
            return
        # The platform boots only images from its registry, so a prebuilt image
        # is imported through a one-line Dockerfile that starts from it.
        with tempfile.TemporaryDirectory(prefix="harbor-unikraft-") as tmp:
            dockerfile = Path(tmp) / DOCKERFILE_NAME
            dockerfile.write_text(f"FROM {self.task_env_config.docker_image}\n")
            await self._run_build(dockerfile, image_ref)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _run_build(self, dockerfile: Path, image_ref: str) -> None:
        """Run ``unikraft build`` and push the result under ``image_ref``."""
        command = [
            self._unikraft_cli,
            "build",
            str(dockerfile),
            "--arch",
            self._arch,
            "-o",
            image_ref,
        ]
        self.logger.debug(f"Building image: {shlex.join(command)}")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timeout = self.task_env_config.build_timeout_sec
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(
                f"Image build timed out after {timeout} seconds: {image_ref}"
            ) from None
        if process.returncode != 0:
            tail = (stderr or stdout)[-_BUILD_LOG_TAIL_BYTES:].decode(errors="replace")
            raise RuntimeError(
                f"Image build failed for {image_ref} (exit {process.returncode}):\n{tail}"
            )

    async def _discard_instance(self, name: str) -> None:
        """Delete the instance a failed create left behind, if it made one."""
        with contextlib.suppress(Exception):
            await self._client().instances.get(name=name).delete()

    async def _stop_detail(self, exc: UnikraftCloudError) -> str | None:
        """Why the platform stopped the instance a failed create left behind.

        The create answers with an item the platform marks failed and leaves
        without a message, so the reason is read off the instance itself.
        """
        uuid = next((error.uuid for error in (exc.errors or []) if error.uuid), None)
        if uuid is None:
            return None
        try:
            instance = await self._client().instances.get(uuid=uuid)
        except Exception:
            return None
        reason, code = instance.stop_reason, instance.stop_code
        if reason == _STOP_REASON_PLATFORM:
            if code == _STOP_CODE_IMAGE_PULL_FAILED:
                return "the node could not pull the image"
            return f"platform stop, code {code}"
        if reason == _STOP_REASON_KERNEL and code is not None:
            return _kernel_stop_detail(code)
        return None if reason is None else f"stop reason {reason}"

    @staticmethod
    def _create_error(
        exc: UnikraftCloudError, name: str, detail: str | None
    ) -> UnikraftCloudError:
        """The create failure, carrying the platform's own reason for it."""
        if detail is None:
            return exc
        return UnikraftCloudError(
            f"Could not create {name}: {detail}",
            kind=exc.kind,
            status=exc.status,
            errors=exc.errors,
            body=exc.body,
        )

    async def _create_instance(self, spec: dict[str, Any], *, attempts: int) -> Any:
        """Create the instance, waiting out a node that cannot pull the image.

        For a few seconds after any push the nodes fail to pull, so a create in
        that window stops at once and still leaves a stopped instance. Every
        failed attempt is read for its reason, then discarded.
        """
        name = spec["name"]
        for attempt in range(1, attempts + 1):
            try:
                return await self._client().instances.create(**spec)
            except UnikraftCloudError as exc:
                detail = await self._stop_detail(exc)
                await self._discard_instance(name)
                if attempt == attempts:
                    raise self._create_error(exc, name, detail) from exc
                self.logger.debug(
                    f"Create attempt {attempt} failed for {name}, retrying in "
                    f"{_IMAGE_RETRY_INTERVAL_SEC}s: {detail or exc}"
                )
                await asyncio.sleep(_IMAGE_RETRY_INTERVAL_SEC)
        raise RuntimeError("unreachable")

    async def _shield_interface(self, shield_uuid: str) -> str:
        """The interface the platform gave the shield, for the task to relay through.

        The platform names it, so nothing a failed attempt leaves behind can
        collide with the next one.
        """
        shield = await self._client().instances.get(uuid=shield_uuid)
        interfaces = shield.network_interfaces or []
        uuid = interfaces[0].uuid if interfaces else None
        if not uuid:
            raise RuntimeError("The network shield came up without an interface.")
        return uuid

    async def _start_shield(self) -> None:
        ukc = self._client()
        shield = await self._create_instance(
            self._shield_spec(), attempts=_IMAGE_RETRY_ATTEMPTS
        )
        if not shield.uuid:
            raise RuntimeError(
                "The platform created the network shield without a UUID."
            )
        self._shield_uuid = shield.uuid
        self._shield_interface_uuid = await self._shield_interface(shield.uuid)
        plugin = ukc.instances.get(uuid=shield.uuid).plugin(SHIELD_PLUGIN_NAME)
        self._shield_api = await plugin.client(_ShieldApi)
        await self._shield_api.wait_ready(self._ready_timeout_sec)
        await self._shield_api.replace_policies(
            self._shield_policies(self.network_policy)
        )

    @override
    async def start(self, force_build: bool) -> None:
        if self._effective_storage_mb:
            self.logger.warning(
                f"storage_mb={self._effective_storage_mb} is not applied on "
                f"{self.type().value}: the root filesystem lives in instance memory."
            )
        ukc = self._client()
        image_ref = self._image_ref()

        async with timed_environment_subphase(self, "image_build"):
            if self._require_prebuilt_image:
                self._assert_prebuilt(image_ref, force_build)
                await self._assert_image_exists(image_ref)
            elif force_build or not await self._image_exists(image_ref):
                await self._build_image(image_ref, force_build)
            else:
                self.logger.debug(f"Using existing image {image_ref}")

        if self._needs_shield():
            async with timed_environment_subphase(self, "shield_provision"):
                await self._start_shield()

        async with timed_environment_subphase(self, "sandbox_provision"):
            instance = await self._create_instance(
                self._instance_spec(image_ref), attempts=_IMAGE_RETRY_ATTEMPTS
            )
            if not instance.uuid:
                raise RuntimeError("The platform created the instance without a UUID.")
            self._instance_uuid = instance.uuid
            self._sandbox = ukc.instances.get(uuid=instance.uuid).sandbox()

        async with timed_environment_subphase(self, "plugin_ready"):
            await self._sandbox.wait_ready(timeout=self._ready_timeout_sec)

        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self._upload_environment_dir_after_start()

    @override
    async def pause(self) -> None:
        """Suspend the instance, keeping its memory and running processes.

        The guest clock stops with it, so an agent's timeouts do not run down
        while it is paused.
        """
        await self._instance_handle("pause").suspend()

    @override
    async def resume(self) -> None:
        """Start the suspended instance and wait for the plugin to answer again."""
        await self._instance_handle("resume").start()
        await self._require_sandbox().wait_ready(timeout=self._ready_timeout_sec)

    def _instance_handle(self, verb: str) -> Any:
        """The running instance, for an operation that needs one."""
        if self._instance_uuid is None:
            raise RuntimeError(f"Cannot {verb} before the environment has started.")
        return self._client().instances.get(uuid=self._instance_uuid)

    @override
    async def _apply_network_policy(self, network_policy: NetworkPolicy) -> None:
        if self._shield_api is None:
            raise RuntimeError(
                "The environment started without a network shield, so its "
                "network policy cannot change. Declare the policy as a phase "
                "policy so the shield is provisioned at start."
            )
        await self._shield_api.replace_policies(self._shield_policies(network_policy))

    async def _remove_instance(
        self, instance_uuid: str, *, delete: bool, label: str
    ) -> None:
        handle = self._client().instances.get(uuid=instance_uuid)
        try:
            if not delete:
                await handle.stop()
                return
            try:
                await handle.delete()
            except UnikraftCloudError:
                # A running instance may need to stop before it can go.
                await handle.stop()
                await handle.delete()
        except Exception as exc:
            verb = "deleting" if delete else "stopping"
            self.logger.error(f"Error {verb} {label} {instance_uuid}: {exc}")

    @override
    async def stop(self, delete: bool) -> None:
        """Stops the environment and optionally deletes it."""
        if self._ukc is None:
            return
        try:
            if self._instance_uuid is not None:
                if not delete:
                    self.logger.warning(
                        "The root filesystem of a stopped Unikraft instance lives "
                        "in memory, so its changes do not survive the stop."
                    )
                await self._remove_instance(
                    self._instance_uuid, delete=delete, label="instance"
                )
            if self._shield_uuid is not None:
                await self._remove_instance(
                    self._shield_uuid, delete=delete, label="network shield"
                )
        finally:
            self._sandbox = None
            self._shield_api = None
            if delete:
                self._instance_uuid = None
                self._shield_uuid = None
            ukc, self._ukc = self._ukc, None
            await ukc.aclose()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _exec_env(self, env: dict[str, str] | None) -> dict[str, str]:
        """The environment a command sees, layered as a Docker exec would.

        The plugin starts every command with no variables at all, so the
        image's defaults and its Dockerfile ENV are supplied here.
        """
        merged = {"PATH": _DOCKER_DEFAULT_PATH, "HOME": "/root"}
        merged.update(self._dockerfile_env)
        merged.update(self._merge_env(env) or {})
        return merged

    def _exec_command(self, command: str, user: str | int | None) -> str:
        user = self._resolve_user(user)
        if user is None:
            user = self._dockerfile_user
        if _is_root(user):
            return command
        assert user is not None
        return _run_as_user(command, user)

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """
        Executes a command in the environment.

        Args:
            command: The command to execute.
            cwd: The working directory in which to execute the command.
            env: The environment variables to set.
            timeout_sec: The timeout in seconds.
            user: Username or UID to run the command as. None falls back to
                ``self.default_user``, then to the Dockerfile's USER.
        """
        sandbox = self._require_sandbox()
        full_command = self._exec_command(command, user)
        exec_env = self._exec_env(env)
        exec_cwd = effective_exec_cwd(
            cwd, self.task_env_config.workdir, self._dockerfile_workdir
        )
        callback = self._output_callback()
        if callback is not None:
            return await self._exec_streaming(
                sandbox, full_command, exec_cwd, exec_env, timeout_sec, callback
            )

        try:
            result = await sandbox.exec(
                full_command,
                cwd=exec_cwd,
                env=exec_env,
                timeout=timeout_sec,
                wait_delay=None if timeout_sec is None else _INTERRUPT_GRACE_SEC,
            )
        except ExecTimeoutError as exc:
            raise RuntimeError(
                f"Command timed out after {timeout_sec} seconds"
            ) from exc
        await self._forget_command(sandbox, result.uuid)
        if result.interrupted:
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        return ExecResult(
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
            return_code=result.exit_code,
        )

    async def _exec_streaming(
        self,
        sandbox: Sandbox,
        command: str,
        cwd: str | None,
        env: dict[str, str],
        timeout_sec: int | None,
        callback: Any,
    ) -> ExecResult:
        """Run a command, handing each output piece to ``callback`` as it arrives."""
        handle = await sandbox.run(command, cwd=cwd, env=env)
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        decoders = {
            stream: codecs.getincrementaldecoder("utf-8")(errors="replace")
            for stream in buffers
        }

        async def consume() -> None:
            chunk: OutputChunk
            async for chunk in handle.stream():
                buffers[chunk.stream].extend(chunk.data)
                text = decoders[chunk.stream].decode(chunk.data)
                if text:
                    await callback(text, chunk.stream)

        try:
            if timeout_sec is None:
                await consume()
            else:
                await asyncio.wait_for(consume(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(handle.signal(2), _INTERRUPT_GRACE_SEC)
            raise RuntimeError(
                f"Command timed out after {timeout_sec} seconds"
            ) from None

        for stream, decoder in decoders.items():
            text = decoder.decode(b"", final=True)
            if text:
                await callback(text, stream)  # type: ignore[arg-type]
        info = await handle.get()
        await self._forget_command(sandbox, handle.uuid)
        if info.exitcode is None:
            raise RuntimeError(f"Command {handle.uuid} ended without an exit code.")
        return ExecResult(
            stdout=bytes(buffers["stdout"]).decode(errors="replace"),
            stderr=bytes(buffers["stderr"]).decode(errors="replace"),
            return_code=info.exitcode,
        )

    async def _forget_command(self, sandbox: Sandbox, command_uuid: str) -> None:
        """Drop a finished command so the plugin does not keep its output."""
        with contextlib.suppress(UnikraftCloudError):
            await sandbox.command(command_uuid).delete()

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    async def _write_remote(self, target_path: str, data: bytes) -> None:
        """Write ``data`` to ``target_path``, creating the directories above it."""
        files = self._require_sandbox().fs
        target = PurePosixPath(target_path)
        first, rest = data[:_UPLOAD_CHUNK_BYTES], data[_UPLOAD_CHUNK_BYTES:]
        # The file lands at the given path and `parents` makes the directories
        # above it; `filename` applies only to a path that is a directory.
        await files.upload(target_path, target.name, first, parents=True)
        for offset in range(0, len(rest), _UPLOAD_CHUNK_BYTES):
            await files.write(
                target_path, rest[offset : offset + _UPLOAD_CHUNK_BYTES], append=True
            )

    async def _remove_remote(self, path: str) -> None:
        await self.exec(f"rm -f {shlex.quote(path)}", user="root")

    def _transfer_path(self, suffix: str) -> str:
        return str(_TRANSFER_DIR / f".hb-transfer-{uuid4().hex}{suffix}")

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """
        Adds a local file to the environment.

        Args:
            source_path: The path to the source local file.
            target_path: The path to which to copy the file.
        """
        await self._write_remote(target_path, Path(source_path).read_bytes())

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """
        Adds a local directory to the environment.

        The tree travels as one tar archive, which keeps modes, symlinks and
        empty directories and fails loudly when it arrives incomplete.

        Args:
            source_dir: The path to the source local directory.
            target_dir: The path to which to copy the directory.
        """
        archive = self._transfer_path(".tar.gz")
        await self._write_remote(
            archive, pack_dir_to_bytes(source_dir, compress=True).getvalue()
        )
        try:
            result = await self.exec(
                remote_unpack_command(archive, target_dir), user="root"
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"Failed to unpack {source_dir} into {target_dir}: {result.stderr}"
                )
        finally:
            await self._remove_remote(archive)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """
        Downloads a file from the environment to the local machine.

        Args:
            source_path: The path to the source file in the environment.
            target_path: The local path to which to copy the file.
        """
        data = await self._require_sandbox().fs.read(source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """
        Downloads a directory from the environment to the local machine. This overwrites
        existing files in the target directory.

        Args:
            source_dir: The path to the source directory in the environment.
            target_dir: The local path to which to copy the directory.
        """
        archive = self._transfer_path(".tar.gz")
        result = await self.exec(remote_pack_command(source_dir, archive), user="root")
        if result.return_code != 0:
            raise RuntimeError(f"Failed to pack {source_dir}: {result.stderr}")
        try:
            data = await self._require_sandbox().fs.read(archive)
        finally:
            await self._remove_remote(archive)
        extract_dir_from_bytes(data, target_dir)
