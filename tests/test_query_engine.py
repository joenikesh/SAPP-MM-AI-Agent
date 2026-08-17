import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "material-api"))

from database import Base  # noqa: E402
from models import Material  # noqa: E402
from query_engine import MaterialQueryRequest, execute_material_query  # noqa: E402


class QueryEngineTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.db.add_all([
            Material(material="M-1", material_description="Small pump", plant="1000", standard_price=90, currency="USD"),
            Material(material="M-2", material_description="Large pump", plant="1000", standard_price=700, currency="USD"),
            Material(material="M-3", material_description="Valve", plant="2000", standard_price=550, currency="USD"),
            Material(material="M-4", material_description="Unpriced pump", plant="1000", standard_price=None, currency="USD"),
        ])
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_cheapest_pump_in_plant(self):
        result = execute_material_query(self.db, MaterialQueryRequest.model_validate({
            "search_text": "pump",
            "filters": [{"field": "plant", "operator": "eq", "value": "1000"}],
            "sort": [{"field": "standard_price", "direction": "asc"}],
            "limit": 1,
        }))
        self.assertEqual(result["items"][0].material, "M-1")

    def test_materials_over_500(self):
        result = execute_material_query(self.db, MaterialQueryRequest.model_validate({
            "filters": [
                {"field": "standard_price", "operator": "gt", "value": 500},
                {"field": "currency", "operator": "eq", "value": "USD"},
            ],
            "sort": [{"field": "standard_price", "direction": "desc"}],
        }))
        self.assertEqual([item.material for item in result["items"]], ["M-2", "M-3"])

    def test_maximum_price_aggregate(self):
        result = execute_material_query(self.db, MaterialQueryRequest.model_validate({
            "aggregate": {"function": "max", "field": "standard_price"}
        }))
        self.assertEqual(result["value"], 700)


if __name__ == "__main__":
    unittest.main()
