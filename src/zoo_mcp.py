"""
Autodistill Server MCP Implementation

Entry point for running the MCP server for the Autodistill system.
This file imports and runs the server from the modular components.

JSON configuration:
{
    "mcpServers": {
        "autodistill-server": {
            "type": "stdio",
            "command": "uv",
            "args": [
                "--directory",
                "<path-to-sensor-mcp>/examples/zoo",
                "run",
                "zoo_mcp.py"
            ]
        }
    }
}
"""

from server import app, mcp

# Run the MCP server when executed directly
if __name__ == "__main__":
    # Run the MCP Server with stdio transport
    mcp.run(transport="stdio")
