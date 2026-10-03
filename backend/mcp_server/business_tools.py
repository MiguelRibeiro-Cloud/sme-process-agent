"""Deterministic Northstar business-system functions.

These functions deliberately contain no LLM or MCP logic.  The MCP server is
their integration boundary; they remain independently testable.
"""

import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "synthetic_company"
CRM_FILE = DATA_DIR / "crm.json"
PRICING_FILE = DATA_DIR / "pricing.json"


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path.name}")
    return records


def crm_get_customer(customer_name: str) -> dict[str, Any] | None:
    """Look up a customer in the company's CRM by customer name."""
    normalized_name = customer_name.strip().casefold()
    for customer in _load_records(CRM_FILE):
        if str(customer.get("name", "")).casefold() == normalized_name:
            return customer
    return None


def pricing_get_product(sku: str) -> dict[str, Any] | None:
    """Look up product details and standard price by SKU."""
    normalized_sku = sku.strip().casefold()
    for product in _load_records(PRICING_FILE):
        if str(product.get("sku", "")).casefold() == normalized_sku:
            return product
    return None
