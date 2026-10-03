"""Northstar Business Systems MCP server entry point."""

from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field

from backend.mcp_server.business_tools import (
    crm_get_customer as get_customer,
    pricing_get_product as get_product,
)


SERVER_NAME = "Northstar Business Systems"
mcp = MCPServer(SERVER_NAME)


@mcp.tool()
def crm_get_customer(
    customer_name: Annotated[str, Field(description="The customer's company name.")],
) -> dict | None:
    """Look up a customer in the company's CRM by customer name."""
    return get_customer(customer_name)


@mcp.tool()
def pricing_get_product(
    sku: Annotated[str, Field(description="The product's stock-keeping unit (SKU).")],
) -> dict | None:
    """Look up product details and standard price by SKU."""
    return get_product(sku)


if __name__ == "__main__":
    mcp.run("stdio")
