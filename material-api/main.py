from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas import MaterialSummary, MaterialDetail
from fastapi import Query
from database import Base, engine, SessionLocal
from models import Material
from schemas import MaterialCreate
from query_engine import MaterialQueryRequest, execute_material_query


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="SAP MM Material API",
    version="0.1.0"
)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {
        "service": "SAP MM Material API",
        "status": "running",
        "version": "0.1.0"
    }


@app.get(
    "/materials",
    response_model=list[MaterialSummary],
    summary="List Materials"
)
def get_materials(db: Session = Depends(get_db)):
    return db.query(Material).all()

@app.get(
    "/materials/search",
    response_model=list[MaterialSummary],
    summary="Search Materials"
)
def search_materials(
    q: str | None = Query(default=None),
    plant: str | None = Query(default=None),
    material_type: str | None = Query(default=None),
    material_group: str | None = Query(default=None),
    db: Session = Depends(get_db)
):
    query = db.query(Material)

    if q:
        search_value = f"%{q}%"

        query = query.filter(
            Material.material_description.ilike(search_value)
            | Material.material.ilike(search_value)
            | Material.material_group.ilike(search_value)
        )

    if plant:
        query = query.filter(Material.plant == plant)

    if material_type:
        query = query.filter(
            Material.material_type == material_type
        )

    if material_group:
        query = query.filter(
            Material.material_group == material_group
        )

    return query.limit(20).all()


@app.post(
    "/materials/query",
    summary="Execute a validated analytical material query",
)
def query_materials(
    request: MaterialQueryRequest,
    db: Session = Depends(get_db),
):
    """Execute only the allowlisted filters, sorts and aggregates."""
    return execute_material_query(db, request)


@app.post("/materials")
def create_material(
    request: MaterialCreate,
    db: Session = Depends(get_db)
):
    # Generate next synthetic material number
    last_material = (
        db.query(Material)
        .order_by(Material.id.desc())
        .first()
    )

    if last_material:
        next_number = last_material.id + 1
    else:
        next_number = 1

    material_number = f"SYN-NEW-{next_number:06d}"

    # Basic duplicate check
    existing = (
        db.query(Material)
        .filter(
            Material.material_description
            == request.material_description
        )
        .filter(
            Material.plant
            == request.plant
        )
        .first()
    )

    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                "A material with the same description "
                "already exists in this plant."
            )
        )

    new_material = Material(
        material=material_number,
        material_description=request.material_description,
        material_type=request.material_type,
        material_group=request.material_group,
        base_unit=request.base_unit,
        plant=request.plant,

        procurement_type=request.procurement_type,
        mrp_type=request.mrp_type,
        valuation_class=request.valuation_class,
        price_control=request.price_control,

        standard_price=request.standard_price,
        currency=request.currency,
    )

    db.add(new_material)
    db.commit()
    db.refresh(new_material)

    return {
        "status": "CREATED",
        "material": new_material
    }

@app.get(
    "/materials/{material_number}",
    response_model=MaterialDetail,
    summary="Get Material Details"
)
def get_material(
    material_number: str,
    db: Session = Depends(get_db)
):

    material = (
        db.query(Material)
        .filter(Material.material == material_number)
        .first()
    )

    if not material:
        raise HTTPException(
            status_code=404,
            detail="Material not found"
        )

    return material
