import os
import tempfile
import unittest
from pathlib import Path

from backend.mcp_server.business_tools import crm_get_customer, pricing_get_product


class BusinessToolTests(unittest.TestCase):
    def test_crm_lookup_is_case_insensitive_and_structured(self):
        customer = crm_get_customer("acme ltd")
        self.assertEqual(customer["customer_id"], "C-1042")
        self.assertEqual(customer["account_owner"], "Sarah Chen")

    def test_pricing_lookup_is_case_insensitive_and_structured(self):
        product = pricing_get_product("nx-440")
        self.assertEqual(product["standard_price"], 1250.0)
        self.assertEqual(product["currency"], "EUR")

    def test_unknown_records_return_none(self):
        self.assertIsNone(crm_get_customer("Unknown Company"))
        self.assertIsNone(pricing_get_product("UNKNOWN-1"))

    def test_lookups_do_not_depend_on_current_working_directory(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                product = pricing_get_product("SV-200")
            finally:
                os.chdir(original_directory)
        self.assertEqual(product["standard_price"], 480.0)


if __name__ == "__main__":
    unittest.main()
