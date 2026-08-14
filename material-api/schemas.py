from pydantic import BaseModel, ConfigDict


class MaterialCreate(BaseModel):
    material_description: str
    material_type: str
    material_group: str
    base_unit: str
    plant: str

    procurement_type: str | None = None
    mrp_type: str | None = None
    valuation_class: str | None = None
    price_control: str | None = None

    standard_price: float | None = None
    currency: str | None = "USD"


class MaterialSummary(BaseModel):
    material: str
    material_type: str | None = None
    material_group: str | None = None
    material_description: str | None = None
    base_unit: str | None = None
    plant: str | None = None
    valuation_class: str | None = None
    standard_price: float | None = None
    currency: str | None = None

    model_config = ConfigDict(from_attributes=True)


class MaterialDetail(BaseModel):
    id: int
    material: str

    material_type: str | None = None
    material_group: str | None = None
    material_description: str | None = None

    base_unit: str | None = None
    old_material_number: str | None = None
    division: str | None = None

    plant: str | None = None
    storage_location: str | None = None
    purchasing_group: str | None = None

    procurement_type: str | None = None
    mrp_type: str | None = None
    mrp_controller: str | None = None
    lot_size: str | None = None

    planned_delivery_time_days: int | None = None
    gr_processing_time_days: int | None = None

    safety_stock: float | None = None
    reorder_point: float | None = None

    valuation_class: str | None = None
    price_control: str | None = None

    standard_price: float | None = None
    moving_average_price: float | None = None
    price_unit: int | None = None
    currency: str | None = None

    gross_weight: float | None = None
    net_weight: float | None = None
    weight_unit: str | None = None

    volume: float | None = None
    volume_unit: str | None = None

    profit_center: str | None = None
    tax_classification: str | None = None
    sales_status: str | None = None

    created_on: str | None = None

    batch_management: str | None = None
    serial_number_profile: str | None = None
    country_of_origin: str | None = None

    model_config = ConfigDict(from_attributes=True)