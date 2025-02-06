from fastapi import FastAPI, Request, Depends, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, ForeignKey, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import uvicorn
from typing import Optional
from fastapi import HTTPException
from datetime import datetime, date, timedelta
from sqlalchemy.exc import IntegrityError

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    paytype = Column(String)

class Work(Base):
    __tablename__ = "works"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    tea_weight = Column(Float, nullable=True)
    tea_location = Column(String, nullable=True)
    other_cost = Column(Float, nullable=True)
    other_location = Column(String, nullable=True)
    advance_amount = Column(Float, nullable=True)
    adjusted_tea_weight = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    work_date = Column(Date, nullable=False, server_default=func.current_date())
    work_description = Column(String, nullable=True)

    def __repr__(self):
        return f"<Work(id={self.id}, user_id={self.user_id}, date={self.work_date})>"

class Factory(Base):
    __tablename__ = "factories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class FactoryTea(Base):
    __tablename__ = "factory_tea"
    id = Column(Integer, primary_key=True, index=True)
    factory_id = Column(Integer, ForeignKey("factories.id"), nullable=False)
    weight_type = Column(String, nullable=False)
    leaves_weight = Column(Float, nullable=False)
    factory_date = Column(Date, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

# Remove or comment out the static files mount if not needed
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Add custom Jinja filters
def format_commas(value):
    return "{:,.0f}".format(float(value)) if float(value) % 1 == 0 else "{:,.2f}".format(float(value))

templates.env.filters["thousands_commas"] = format_commas

@app.get("/")
async def read_root():
    return RedirectResponse(url="/dashboard")

# Add ping endpoint
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/users")
async def users_page(request: Request, db: Session = Depends(get_db)):
    users = db.query(User).all()
    return templates.TemplateResponse("users.html", {"request": request, "users": users})

@app.post("/users")
async def create_user(
    request: Request,
    username: str = Form(...),
    paytype: str = Form(...),
    db: Session = Depends(get_db)
):
    user = User(username=username, paytype=paytype)
    db.add(user)
    db.commit()
    return RedirectResponse(url="/users", status_code=303)

@app.get("/add-work")
async def add_work_page(
    request: Request, 
    db: Session = Depends(get_db),
    page: int = 1,
    limit: int = 10
):
    # Calculate offset
    offset = (page - 1) * limit
    
    # Get paginated works and total count
    works = db.query(Work).offset(offset).limit(limit).all()
    total = db.query(Work).count()
    
    return templates.TemplateResponse("add_work.html", {
        "request": request,
        "users": db.query(User).all(),
        "works": works,
        "current_page": page,
        "total_pages": (total + limit - 1) // limit,
        "limit": limit,
        "factories": db.query(Factory).all()
    })

@app.post("/add-work")
async def create_work(
    request: Request,
    user_id: int = Form(...),
    tea_weight: Optional[str] = Form(None),
    tea_location: Optional[str] = Form(None),
    other_cost: Optional[str] = Form(None),
    other_location: Optional[str] = Form(None),
    advance_amount: Optional[str] = Form(None),
    work_date: Optional[str] = Form(None),
    work_description: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    def parse_float(value: Optional[str]) -> Optional[float]:
        if value and value.strip():
            try:
                return float(value)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid number value: {value}")
        return None

    tea_weight_float = parse_float(tea_weight)
    other_cost_float = parse_float(other_cost)
    advance_amount_float = parse_float(advance_amount)

    # Handle missing work date
    if not work_date:
        work_date_obj = date.today()
    else:
        try:
            work_date_obj = date.fromisoformat(work_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")

    # Only calculate adjusted tea weight if tea weight is provided
    adjusted_tea = None
    if tea_weight_float is not None:
        if tea_weight_float > 60:
            adjusted_tea = tea_weight_float - 4
        elif tea_weight_float > 25:
            adjusted_tea = tea_weight_float - 3
        elif tea_weight_float > 18:
            adjusted_tea = tea_weight_float - 2
        elif tea_weight_float > 10:
            adjusted_tea = tea_weight_float - 1
        else:
            adjusted_tea = tea_weight_float

    # Create new work record
    work = Work(
        user_id=user_id,
        tea_weight=tea_weight_float,
        tea_location=tea_location,
        other_cost=other_cost_float,
        other_location=other_location,
        advance_amount=advance_amount_float,
        work_date=work_date_obj,
        adjusted_tea_weight=adjusted_tea if tea_weight_float is not None else None,
        work_description=work_description
    )
    
    db.add(work)
    db.commit()
    
    return RedirectResponse(url="/add-work", status_code=303)

@app.get("/tea-leaves")
async def tea_leaves_report(request: Request, db: Session = Depends(get_db)):
    results = db.query(
        Work.work_date,
        User.username,
        Work.tea_location,
        Work.tea_weight,
        Work.adjusted_tea_weight,
    ).join(User).filter(
        Work.tea_weight > 0
    ).order_by(
        Work.work_date.desc(),
        User.username
    ).all()

    return templates.TemplateResponse("tea_leaves.html", {
        "request": request,
        "records": results
    })

@app.get("/other-work")
async def other_work_report(request: Request, db: Session = Depends(get_db)):
    results = db.query(
        Work.work_date,
        User.username,
        Work.other_location,
        Work.work_description,
        func.sum(Work.other_cost).label('total_cost')
    ).join(User).filter(
        Work.other_cost > 0
    ).group_by(
        Work.user_id,
        Work.work_date,
        Work.other_location,
        Work.work_description
    ).order_by(
        Work.work_date.desc(),
        User.username
    ).all()

    return templates.TemplateResponse("other_work.html", {
        "request": request,
        "records": results
    })

@app.get("/dashboard")
async def dashboard(
    request: Request, 
    db: Session = Depends(get_db),
    year: int = None,
    month: int = None
):
    # Default to current month/year if not provided
    now = datetime.now()
    selected_year = year or now.year
    selected_month = month or now.month
    
    # Get monthly tea leaves total
    monthly_total = db.query(
        func.sum(Work.adjusted_tea_weight)
    ).filter(
        func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
        func.strftime('%m', Work.work_date) == f"{selected_month:02d}",
        Work.adjusted_tea_weight > 0
    ).scalar() or 0.0

    # Get user-wise totals for the month
    user_totals = db.query(
        User.username,
        func.sum(Work.tea_weight).label('user_total')
    ).join(Work).filter(
        func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
        func.strftime('%m', Work.work_date) == f"{selected_month:02d}",
        Work.tea_weight > 0
    ).group_by(User.id).all()

    # Prepare chart data
    user_labels = [user.username for user in user_totals]
    user_data = [float(user.user_total) for user in user_totals]

    # Generate year options (2020 to current year)
    current_year = now.year
    years = list(range(2020, current_year + 1))
    
    # Get factory tea totals
    factory_tea_totals = db.query(
        Factory.name,
        func.sum(FactoryTea.leaves_weight).label('total_weight')
    ).join(FactoryTea).filter(
        func.strftime('%Y', FactoryTea.factory_date) == f"{selected_year:04d}",
        func.strftime('%m', FactoryTea.factory_date) == f"{selected_month:02d}"
    ).group_by(Factory.name).all()

    total_factory_weight = db.query(
        func.sum(FactoryTea.leaves_weight)
    ).filter(
        func.strftime('%Y', FactoryTea.factory_date) == f"{selected_year:04d}",
        func.strftime('%m', FactoryTea.factory_date) == f"{selected_month:02d}"
    ).scalar() or 0.0

    # Get 30-day tea leaves data
    end_date = date.today()
    start_date = end_date - timedelta(days=29)
    
    daily_tea_totals = db.query(
        func.date(Work.work_date).label('date'),
        Work.tea_location,
        func.sum(Work.adjusted_tea_weight).label('total')
    ).filter(
        Work.work_date >= start_date,
        Work.work_date <= end_date,
        Work.adjusted_tea_weight > 0
    ).group_by(func.date(Work.work_date), Work.tea_location).all()

    # Create complete date range with zeros
    date_range = [start_date + timedelta(days=x) for x in range(30)]
    daily_labels = [d.strftime('%Y-%m-%d') for d in date_range]

    locations = list({result.tea_location for result in daily_tea_totals if result.tea_location})
    location_data = {loc: [0]*30 for loc in locations}

    # Fill the location data
    for result in daily_tea_totals:
        idx = (datetime.strptime(result.date, '%Y-%m-%d').date() - start_date).days
        if 0 <= idx < 30:
            location_data[result.tea_location][idx] += float(result.total)

    # Convert to chart.js format
    location_datasets = []
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
    for i, (loc, values) in enumerate(location_data.items()):
        location_datasets.append({
            'label': loc or 'Unknown Location',
            'data': values,
            'backgroundColor': colors[i % len(colors)] + '77',  # Add alpha channel
            'borderColor': colors[i % len(colors)],
            'borderWidth': 1
        })

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "monthly_total": monthly_total,
        "user_totals": user_totals,
        "user_labels": user_labels,
        "user_data": user_data,
        "years": years,
        "selected_year": selected_year,
        "selected_month": selected_month,
        "total_factory_weight": total_factory_weight,
        "factory_tea_totals": factory_tea_totals,
        "daily_labels": daily_labels,
        "location_datasets": location_datasets
    })

@app.get("/salary")
async def salary_report(
    request: Request,
    db: Session = Depends(get_db),
    year: int = None,
    month: int = None,
    price_per_kg: float = 48.214
):
    # Default to current month/year if not provided
    now = datetime.now()
    selected_year = year or now.year
    selected_month = month or now.month
    
    # Get all users
    users = db.query(User).all()
    
    # Get salary data for each user
    salary_data = []
    for user in users:
        # Get tea leaves total
        tea_total = db.query(
            func.sum(Work.adjusted_tea_weight)
        ).filter(
            Work.user_id == user.id,
            func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
            func.strftime('%m', Work.work_date) == f"{selected_month:02d}",
            Work.adjusted_tea_weight > 0
        ).scalar() or 0.0
        
        # Get other work total
        other_total = db.query(
            func.sum(Work.other_cost)
        ).filter(
            Work.user_id == user.id,
            func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
            func.strftime('%m', Work.work_date) == f"{selected_month:02d}",
            Work.other_cost > 0
        ).scalar() or 0.0
        
        # Get advance total
        advance_total = db.query(
            func.sum(Work.advance_amount)
        ).filter(
            Work.user_id == user.id,
            func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
            func.strftime('%m', Work.work_date) == f"{selected_month:02d}",
            Work.advance_amount > 0
        ).scalar() or 0.0
        
        # Calculate values
        tea_income = tea_total * price_per_kg
        total_salary = tea_income + other_total
        balance = total_salary - advance_total
        
        salary_data.append({
            "user": user,
            "tea_income": tea_income,
            "other_income": other_total,
            "total_salary": total_salary,
            "advance": advance_total,
            "balance": balance
        })
    
    return templates.TemplateResponse("salary.html", {
        "request": request,
        "salary_data": salary_data,
        "price_per_kg": price_per_kg,
        "selected_year": selected_year,
        "selected_month": selected_month,
        "years": list(range(2020, now.year + 1))
    })

@app.get("/factories")
async def factories_page(request: Request, db: Session = Depends(get_db)):
    factories = db.query(Factory).order_by(Factory.created_at.desc()).all()
    return templates.TemplateResponse("factories.html", {
        "request": request,
        "factories": factories
    })

@app.post("/factories")
async def create_factory(
    request: Request,
    name: str = Form(...),
    db: Session = Depends(get_db)
):
    factory = Factory(name=name)
    db.add(factory)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Factory name already exists")
    return RedirectResponse(url="/factories", status_code=303)

@app.post("/add-factory-tea")
async def create_factory_tea(
    request: Request,
    factory_id: int = Form(...),
    factory_date: str = Form(...),
    weight_type: str = Form(...),
    leaves_weight: str = Form(...),
    db: Session = Depends(get_db)
):
    # Parse numeric field
    try:
        weight = float(leaves_weight)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid weight value")

    # Parse date
    try:
        date_obj = date.fromisoformat(factory_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    # Create factory tea record
    factory_tea = FactoryTea(
        factory_id=factory_id,
        weight_type=weight_type,
        leaves_weight=weight,
        factory_date=date_obj
    )
    
    db.add(factory_tea)
    db.commit()
    
    return RedirectResponse(url="/add-work", status_code=303)

# Add server runner
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
