# Copyright 2026 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration schema for kubeflow-mcp.

Supports configuration from:
1. Config file: ~/.kubeflow-mcp.yaml
2. Environment variables (override config file)
3. CLI arguments (override both)

Example config file (~/.kubeflow-mcp.yaml):

.. code-block:: yaml

        server:
            clients:
                - trainer
                - optimizer
            persona: ml-engineer
            namespaces:
                - ml-team-dev
                - ml-team-prod
            transport: stdio

        trainer:
            default_runtime: torch-distributed

        logging:
            level: INFO
            format: json
"""

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Config file locations (searched in order)
CONFIG_PATHS = [
    Path.home() / ".kubeflow-mcp.yaml",
    Path.home() / ".kubeflow-mcp.yml",
    Path.home() / ".config" / "kubeflow-mcp" / "config.yaml",
    Path.cwd() / ".kubeflow-mcp.yaml",
]


class ServerConfig(BaseModel):
    """Server configuration."""

    clients: list[str] = Field(default=["trainer"])
    persona: str = Field(default="ml-engineer")
    namespaces: list[str] | None = None
    transport: str = Field(default="stdio")


class TrainerConfig(BaseModel):
    """Trainer-specific configuration."""

    default_runtime: str | None = None
    default_namespace: str | None = None


class OptimizerConfig(BaseModel):
    """Optimizer-specific configuration."""

    default_algorithm: str = Field(default="random")
    max_trials: int = Field(default=10)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str | None = Field(default=None)


class Config(BaseModel):
    """Root configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _find_config_file() -> Path | None:
    """Find the first existing config file."""
    for path in CONFIG_PATHS:
        if path.exists():
            return path
    return None


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load config from YAML file."""
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except ImportError:
        logger.warning("PyYAML not installed, skipping config file")
        return {}
    except Exception as e:
        logger.warning(f"Failed to load config from {path}: {e}")
        return {}


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file and environment variables.

    Priority (highest to lowest):
    1. Environment variables
    2. Specified config file
    3. Default config file locations
    4. Default values

    Args:
        config_path: Optional explicit path to config file

    Returns:
        Merged configuration
    """
    # Start with defaults
    file_config: dict[str, Any] = {}

    # Load from file
    if config_path and config_path.exists():
        file_config = _load_yaml_config(config_path)
        logger.debug(f"Loaded config from {config_path}")
    else:
        default_path = _find_config_file()
        if default_path:
            file_config = _load_yaml_config(default_path)
            logger.debug(f"Loaded config from {default_path}")

    # Build server config with env overrides
    server_file = file_config.get("server", {})
    server = ServerConfig(
        clients=os.getenv(
            "KUBEFLOW_MCP_CLIENTS",
            ",".join(server_file.get("clients", ["trainer"])),
        ).split(","),
        persona=os.getenv(
            "KUBEFLOW_MCP_PERSONA",
            server_file.get("persona", "ml-engineer"),
        ),
        namespaces=server_file.get("namespaces"),
        transport=os.getenv(
            "MCP_TRANSPORT",
            server_file.get("transport", "stdio"),
        ),
    )

    # Build logging config with env overrides
    logging_file = file_config.get("logging", {})
    logging_config = LoggingConfig(
        level=os.getenv("LOG_LEVEL", logging_file.get("level", "INFO")),
        format=os.getenv("LOG_FORMAT", logging_file.get("format")),
    )

    # Build client-specific configs
    trainer_file = file_config.get("trainer", {})
    trainer = TrainerConfig(
        default_runtime=trainer_file.get("default_runtime"),
        default_namespace=trainer_file.get("default_namespace"),
    )

    optimizer_file = file_config.get("optimizer", {})
    optimizer = OptimizerConfig(
        default_algorithm=optimizer_file.get("default_algorithm", "random"),
        max_trials=optimizer_file.get("max_trials", 10),
    )

    return Config(
        server=server,
        trainer=trainer,
        optimizer=optimizer,
        logging=logging_config,
    )


def get_config_path() -> Path | None:
    """Get the path to the active config file (if any)."""
    return _find_config_file()
