"""Validation for LLM-produced, read-only Material Master query plans."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


TextField = Literal[
    "material", "material_description", "material_type", "material_group",
    "base_unit", "old_material_number", "division", "plant",
    "storage_location", "purchasing_group",
    "procurement_type", "mrp_type", "valuation_class", "price_control",
    "mrp_controller", "lot_size", "currency", "weight_unit", "volume_unit",
    "profit_center", "tax_classification", "sales_status", "created_on",
    "batch_management", "serial_number_profile", "country_of_origin",
]
NumericField = Literal[
    "standard_price", "moving_average_price", "price_unit", "safety_stock",
    "reorder_point", "planned_delivery_time_days", "gr_processing_time_days",
    "gross_weight", "net_weight", "volume",
]
QueryField = TextField | NumericField

TEXT_FIELDS = {
    "material", "material_description", "material_type", "material_group",
    "base_unit", "old_material_number", "division", "plant",
    "storage_location", "purchasing_group",
    "procurement_type", "mrp_type", "valuation_class", "price_control",
    "mrp_controller", "lot_size", "currency", "weight_unit", "volume_unit",
    "profit_center", "tax_classification", "sales_status", "created_on",
    "batch_management", "serial_number_profile", "country_of_origin",
}
NUMERIC_FIELDS = {
    "standard_price", "moving_average_price", "price_unit", "safety_stock",
    "reorder_point", "planned_delivery_time_days", "gr_processing_time_days",
    "gross_weight", "net_weight", "volume",
}


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: QueryField
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains", "in"]
    value: Any

    @model_validator(mode="after")
    def validate_field_operator(self):
        if self.operator in {"gt", "gte", "lt", "lte"}:
            if self.field not in NUMERIC_FIELDS:
                raise ValueError("comparison operators require a numeric field")
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError("numeric comparisons require a number")
        if self.operator == "contains" and self.field not in TEXT_FIELDS:
            raise ValueError("contains requires a text field")
        if self.operator == "in" and (
            not isinstance(self.value, list) or not self.value or len(self.value) > 20
        ):
            raise ValueError("in requires a non-empty list of at most 20 values")
        return self


class Sort(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: QueryField
    direction: Literal["asc", "desc"] = "asc"


class Aggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    function: Literal["count", "min", "max", "avg", "sum"]
    field: QueryField | None = None
    group_by: TextField | None = None

    @model_validator(mode="after")
    def validate_aggregate(self):
        if self.function == "count":
            return self
        if self.field not in NUMERIC_FIELDS:
            raise ValueError("min, max, avg and sum require a numeric field")
        return self


class MaterialQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search_text: str | None = Field(default=None, max_length=100)
    filters: list[Filter] = Field(default_factory=list, max_length=12)
    sort: list[Sort] = Field(default_factory=list, max_length=3)
    limit: int = Field(default=20, ge=1, le=100)
    aggregate: Aggregate | None = None

    @field_validator("search_text")
    @classmethod
    def clean_search_text(cls, value):
        if value is None:
            return None
        value = value.strip()
        return value or None


def validate_query_plan(raw: Any) -> dict:
    """Default-deny invalid or model-invented fields/operators."""
    try:
        return MaterialQueryPlan.model_validate(raw).model_dump(exclude_none=True)
    except ValidationError as exc:
        raise ValueError(f"Invalid material query plan: {exc}") from exc
