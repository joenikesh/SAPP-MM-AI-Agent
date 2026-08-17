"""Deterministic compiler for the allowlisted Material Master query language."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import asc, desc, func, or_
from sqlalchemy.orm import Query

from models import Material


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
QUERY_FIELDS = TEXT_FIELDS | NUMERIC_FIELDS


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    operator: str
    value: Any

    @model_validator(mode="after")
    def allowed(self):
        if self.field not in QUERY_FIELDS:
            raise ValueError("field is not queryable")
        if self.operator not in {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "in"}:
            raise ValueError("operator is not allowed")
        if self.operator in {"gt", "gte", "lt", "lte"} and self.field not in NUMERIC_FIELDS:
            raise ValueError("comparison requires a numeric field")
        if self.operator in {"gt", "gte", "lt", "lte"} and (
            isinstance(self.value, bool) or not isinstance(self.value, (int, float))
        ):
            raise ValueError("numeric comparison requires a number")
        if self.operator == "contains" and self.field not in TEXT_FIELDS:
            raise ValueError("contains requires a text field")
        if self.operator == "in" and (
            not isinstance(self.value, list) or not self.value or len(self.value) > 20
        ):
            raise ValueError("in requires 1-20 values")
        return self


class SortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    direction: str = "asc"

    @model_validator(mode="after")
    def allowed(self):
        if self.field not in QUERY_FIELDS or self.direction not in {"asc", "desc"}:
            raise ValueError("invalid sort")
        return self


class AggregateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    function: str
    field: str | None = None
    group_by: str | None = None

    @model_validator(mode="after")
    def allowed(self):
        if self.function not in {"count", "min", "max", "avg", "sum"}:
            raise ValueError("invalid aggregate")
        if self.function != "count" and self.field not in NUMERIC_FIELDS:
            raise ValueError("aggregate requires a numeric field")
        if self.field is not None and self.field not in QUERY_FIELDS:
            raise ValueError("invalid aggregate field")
        if self.group_by is not None and self.group_by not in TEXT_FIELDS:
            raise ValueError("invalid group field")
        return self


class MaterialQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search_text: str | None = Field(default=None, max_length=100)
    filters: list[FilterSpec] = Field(default_factory=list, max_length=12)
    sort: list[SortSpec] = Field(default_factory=list, max_length=3)
    limit: int = Field(default=20, ge=1, le=100)
    aggregate: AggregateSpec | None = None


def apply_filters(query: Query, plan: MaterialQueryRequest) -> Query:
    if plan.search_text:
        pattern = f"%{plan.search_text}%"
        query = query.filter(or_(
            Material.material_description.ilike(pattern),
            Material.material.ilike(pattern),
            Material.material_group.ilike(pattern),
        ))
    operations = {
        "eq": lambda column, value: column == value,
        "ne": lambda column, value: column != value,
        "gt": lambda column, value: column > value,
        "gte": lambda column, value: column >= value,
        "lt": lambda column, value: column < value,
        "lte": lambda column, value: column <= value,
        "contains": lambda column, value: column.ilike(f"%{value}%"),
        "in": lambda column, value: column.in_(value),
    }
    for item in plan.filters:
        query = query.filter(operations[item.operator](getattr(Material, item.field), item.value))
    return query


def execute_material_query(db, plan: MaterialQueryRequest) -> dict:
    query = apply_filters(db.query(Material), plan)
    if plan.aggregate:
        spec = plan.aggregate
        target = getattr(Material, spec.field) if spec.field else Material.id
        expression = func.count(target) if spec.function == "count" else getattr(func, spec.function)(target)
        if spec.group_by:
            group = getattr(Material, spec.group_by)
            rows = query.with_entities(group, expression).group_by(group).limit(plan.limit).all()
            return {"kind": "aggregate", "aggregate": spec.model_dump(), "rows": [
                {spec.group_by: row[0], "value": row[1]} for row in rows
            ]}
        value = query.with_entities(expression).scalar()
        return {"kind": "aggregate", "aggregate": spec.model_dump(), "value": value}

    for item in plan.sort:
        column = getattr(Material, item.field)
        if item.field in NUMERIC_FIELDS:
            query = query.filter(column.is_not(None))
        query = query.order_by(desc(column) if item.direction == "desc" else asc(column))
    # Stable tie-breaker makes results repeatable across SAP/database adapters.
    query = query.order_by(asc(Material.material))
    items = query.limit(plan.limit).all()
    return {"kind": "records", "count": len(items), "items": items}
