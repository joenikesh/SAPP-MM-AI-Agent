from pathlib import Path

import pandas as pd

from database import Base, engine, SessionLocal
from models import Material


BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "synthetic_sap_material_master_200.csv"


def clean_value(value):
    if pd.isna(value):
        return None

    return value


def seed_database():
    print(f"Reading CSV from: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    try:
        inserted = 0
        skipped = 0

        for _, row in df.iterrows():

            material_number = str(row["Material"]).strip()

            existing = (
                db.query(Material)
                .filter(Material.material == material_number)
                .first()
            )

            if existing:
                skipped += 1
                continue

            material = Material(
                material=material_number,

                material_type=clean_value(row["MaterialType"]),
                material_group=clean_value(row["MaterialGroup"]),
                material_description=clean_value(row["MaterialDescription"]),

                base_unit=clean_value(row["BaseUnit"]),
                old_material_number=clean_value(row["OldMaterialNumber"]),
                division=clean_value(row["Division"]),

                plant=clean_value(row["Plant"]),
                storage_location=clean_value(row["StorageLocation"]),
                purchasing_group=clean_value(row["PurchasingGroup"]),

                procurement_type=clean_value(row["ProcurementType"]),
                mrp_type=clean_value(row["MRPType"]),
                mrp_controller=clean_value(row["MRPController"]),
                lot_size=clean_value(row["LotSize"]),

                planned_delivery_time_days=clean_value(
                    row["PlannedDeliveryTimeDays"]
                ),

                gr_processing_time_days=clean_value(
                    row["GRProcessingTimeDays"]
                ),

                safety_stock=clean_value(row["SafetyStock"]),
                reorder_point=clean_value(row["ReorderPoint"]),

                valuation_class=clean_value(row["ValuationClass"]),
                price_control=clean_value(row["PriceControl"]),

                standard_price=clean_value(row["StandardPrice"]),
                moving_average_price=clean_value(
                    row["MovingAveragePrice"]
                ),

                price_unit=clean_value(row["PriceUnit"]),
                currency=clean_value(row["Currency"]),

                gross_weight=clean_value(row["GrossWeight"]),
                net_weight=clean_value(row["NetWeight"]),
                weight_unit=clean_value(row["WeightUnit"]),

                volume=clean_value(row["Volume"]),
                volume_unit=clean_value(row["VolumeUnit"]),

                profit_center=clean_value(row["ProfitCenter"]),
                tax_classification=clean_value(
                    row["TaxClassification"]
                ),

                sales_status=clean_value(row["SalesStatus"]),
                created_on=clean_value(row["CreatedOn"]),

                batch_management=clean_value(row["BatchManagement"]),
                serial_number_profile=clean_value(
                    row["SerialNumberProfile"]
                ),

                country_of_origin=clean_value(
                    row["CountryOfOrigin"]
                )
            )

            db.add(material)
            inserted += 1

        db.commit()

        print(f"Inserted: {inserted}")
        print(f"Skipped existing: {skipped}")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    seed_database()