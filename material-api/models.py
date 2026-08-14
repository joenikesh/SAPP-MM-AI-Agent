from sqlalchemy import Column, Integer, String, Float
from database import Base


class Material(Base):
    __tablename__ = "materials"

    id = Column(Integer, primary_key=True, index=True)

    material = Column(String, unique=True, index=True, nullable=False)
    material_type = Column(String)
    material_group = Column(String)
    material_description = Column(String, index=True)

    base_unit = Column(String)
    old_material_number = Column(String)
    division = Column(String)

    plant = Column(String, index=True)
    storage_location = Column(String)
    purchasing_group = Column(String)

    procurement_type = Column(String)
    mrp_type = Column(String)
    mrp_controller = Column(String)
    lot_size = Column(String)

    planned_delivery_time_days = Column(Integer)
    gr_processing_time_days = Column(Integer)

    safety_stock = Column(Float)
    reorder_point = Column(Float)

    valuation_class = Column(String)
    price_control = Column(String)

    standard_price = Column(Float)
    moving_average_price = Column(Float)
    price_unit = Column(Integer)
    currency = Column(String)

    gross_weight = Column(Float)
    net_weight = Column(Float)
    weight_unit = Column(String)

    volume = Column(Float)
    volume_unit = Column(String)

    profit_center = Column(String)
    tax_classification = Column(String)
    sales_status = Column(String)

    created_on = Column(String)

    batch_management = Column(String)
    serial_number_profile = Column(String)
    country_of_origin = Column(String)