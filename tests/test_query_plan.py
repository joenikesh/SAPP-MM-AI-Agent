import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-api"))

from query_plan import validate_query_plan  # noqa: E402


class QueryPlanValidationTests(unittest.TestCase):
    def test_cheapest_pump_plan(self):
        plan = validate_query_plan({
            "search_text": "pump",
            "filters": [{"field": "plant", "operator": "eq", "value": "1000"}],
            "sort": [{"field": "standard_price", "direction": "asc"}],
            "limit": 1,
        })
        self.assertEqual(plan["limit"], 1)
        self.assertEqual(plan["sort"][0]["field"], "standard_price")

    def test_rejects_unknown_field(self):
        with self.assertRaises(ValueError):
            validate_query_plan({
                "filters": [{"field": "password", "operator": "eq", "value": "x"}]
            })

    def test_rejects_sql_and_excessive_limit(self):
        with self.assertRaises(ValueError):
            validate_query_plan({"sql": "DELETE FROM materials", "limit": 1000})

    def test_rejects_text_numeric_comparison(self):
        with self.assertRaises(ValueError):
            validate_query_plan({
                "filters": [{"field": "plant", "operator": "gt", "value": 1000}]
            })

    def test_accepts_every_material_master_business_field(self):
        text_fields = [
            "material", "material_description", "material_type", "material_group",
            "base_unit", "old_material_number", "division", "plant",
            "storage_location", "purchasing_group", "procurement_type", "mrp_type",
            "mrp_controller", "lot_size", "valuation_class", "price_control",
            "currency", "weight_unit", "volume_unit", "profit_center",
            "tax_classification", "sales_status", "created_on", "batch_management",
            "serial_number_profile", "country_of_origin",
        ]
        numeric_fields = [
            "standard_price", "moving_average_price", "price_unit", "safety_stock",
            "reorder_point", "planned_delivery_time_days", "gr_processing_time_days",
            "gross_weight", "net_weight", "volume",
        ]
        for field in text_fields:
            plan = validate_query_plan({
                "filters": [{"field": field, "operator": "eq", "value": "x"}]
            })
            self.assertEqual(plan["filters"][0]["field"], field)
        for field in numeric_fields:
            plan = validate_query_plan({
                "filters": [{"field": field, "operator": "gte", "value": 0}]
            })
            self.assertEqual(plan["filters"][0]["field"], field)


if __name__ == "__main__":
    unittest.main()
