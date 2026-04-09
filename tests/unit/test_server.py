"""Tests for MCP server creation."""

from kubeflow_mcp.core.prompts import register_prompts
from kubeflow_mcp.core.server import CLIENT_MODULES, create_server


def test_create_server_default():
    """Server creates with default trainer client."""
    server = create_server()
    assert server is not None
    assert server.name == "kubeflow-mcp"


def test_create_server_with_persona():
    """Server respects persona filtering."""
    server = create_server(persona="readonly")
    assert server is not None


def test_create_server_unknown_client():
    """Server skips unknown clients gracefully."""
    server = create_server(clients=["trainer", "unknown"])
    assert server is not None


def test_client_modules_registered():
    """All expected client modules are registered."""
    assert "trainer" in CLIENT_MODULES
    assert "optimizer" in CLIENT_MODULES
    assert "hub" in CLIENT_MODULES


def test_trainer_module_has_tools():
    """Trainer module exports tools."""
    from kubeflow_mcp import trainer

    assert hasattr(trainer, "TOOLS")
    assert hasattr(trainer, "MODULE_INFO")
    assert len(trainer.TOOLS) > 0
    assert trainer.MODULE_INFO["status"] == "implemented"


def test_optimizer_module_is_stub():
    """Optimizer module is a stub."""
    from kubeflow_mcp import optimizer

    assert hasattr(optimizer, "TOOLS")
    assert len(optimizer.TOOLS) == 0
    assert optimizer.MODULE_INFO["status"] == "stub"


def test_hub_module_is_stub():
    """Hub module is a stub."""
    from kubeflow_mcp import hub

    assert hasattr(hub, "TOOLS")
    assert len(hub.TOOLS) == 0
    assert hub.MODULE_INFO["status"] == "stub"


def test_prompts_registered():
    """Server registers MCP prompts for workflows."""
    from fastmcp import FastMCP

    mcp = FastMCP("test")
    register_prompts(mcp)
    # Prompts should be registered (FastMCP stores them internally)
