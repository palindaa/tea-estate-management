from fastapi import FastAPI, Request, Depends, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, ForeignKey, Date, case
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import uvicorn
from typing import Optional
from fastapi import HTTPException
from datetime import datetime, date, timedelta
from sqlalchemy.exc import IntegrityError
from calendar import monthcalendar
from xhtml2pdf import pisa
from fastapi.responses import StreamingResponse
import io
from collections import defaultdict

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    paytype = Column(String)
    basic_salary = Column(Float, nullable=True, default=0.0)
    weekly_salary = Column(Float, nullable=True, default=0.0)

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
    paytype: list[str] = Form(...),
    basic_salary: str = Form(""),
    weekly_salary: str = Form(""),
    db: Session = Depends(get_db)
):
    # Handle empty salary inputs
    def parse_salary(value: str) -> float:
        try:
            return float(value) if value.strip() else 0.0
        except ValueError:
            return 0.0

    paytype_str = ",".join(paytype)
    user = User(
        username=username,
        paytype=paytype_str,
        basic_salary=parse_salary(basic_salary),
        weekly_salary=parse_salary(weekly_salary)
    )
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

def week_of_month(dt):
    first_day = dt.replace(day=1)
    return (dt.isocalendar()[1] - first_day.isocalendar()[1])
    
@app.get("/salary")
async def salary_report(
    request: Request,
    db: Session = Depends(get_db),
    year: int = None,
    month: int = None,
    week: str = None,
    price_per_kg: float = 48.214
):
    # Default to current month/year if not provided
    now = datetime.now()
    selected_year = year or now.year
    selected_month = month or now.month
    selected_week = (int(week) if week is not None else week_of_month(now))
    
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
        
        # Count tea work days
        tea_days = db.query(
            func.count(func.distinct(Work.work_date))
        ).filter(
            Work.user_id == user.id,
            func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
            func.strftime('%m', Work.work_date) == f"{selected_month:02d}",
            Work.tea_weight > 0
        ).scalar() or 0

        # Count extra work days
        extra_days = db.query(
            func.count(func.distinct(Work.work_date))
        ).filter(
            Work.user_id == user.id,
            func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
            func.strftime('%m', Work.work_date) == f"{selected_month:02d}",
            Work.other_cost > 0
        ).scalar() or 0

        # Calculate values
        tea_income = tea_total * price_per_kg
        total_salary = tea_income + other_total

        # Calculate number of Fridays in the selected month
        def count_fridays_in_month(year, month):
            return sum(1 for week in monthcalendar(year, month) if week[4] != 0)

        fridays_count = count_fridays_in_month(selected_year, selected_month)
        adjusted_basic = (user.basic_salary or 0) + (fridays_count * (user.weekly_salary or 0))
        balance = adjusted_basic + total_salary - advance_total
        salary_data.append({
            "user": user,
            "tea_weight": tea_total,
            "tea_income": tea_income,
            "other_income": other_total,
            "total_salary": total_salary,
            "advance": advance_total,
            "balance": balance,
            "adjusted_basic": adjusted_basic,
            "tea_days": tea_days,
            "extra_days": extra_days
        })

    # Process weekly employees with week filter
    weekly_salary_data = []
    for user in users:
        if 'Weekly' not in user.paytype:
            continue

        # Base query
        query = db.query(Work).filter(
            Work.user_id == user.id,
            func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
            func.strftime('%m', Work.work_date) == f"{selected_month:02d}"
        )

        # Apply week filter
        week_start = (selected_week - 1) * 7 + 1
        week_end = week_start + 6
        query = query.filter(
            func.strftime('%d', Work.work_date).between(f"{week_start:02d}", f"{week_end:02d}")
        )

        # Calculate totals
        advance_total = query.with_entities(func.sum(Work.advance_amount)).scalar() or 0.0
        work_days = query.count()

        print('Work', work_days, advance_total, user.weekly_salary, week)
        weekly_salary_data.append({
            "user": user,
            "advance": advance_total,
            "work_days": work_days,
            "balance": (user.weekly_salary or 0) - advance_total
        })

    # Add current datetime to context
    context = {
        "request": request,
        "salary_data": salary_data,
        "price_per_kg": price_per_kg,
        "selected_year": selected_year,
        "selected_month": selected_month,
        "years": list(range(2020, now.year + 1)),
        "users": users,
        "fridays_count": fridays_count,
        "now": now,
        "selected_week": selected_week,
        "weekly_salary_data": weekly_salary_data
    }

    return templates.TemplateResponse("salary.html", context)

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

@app.get("/advances")
async def advances_page(
    request: Request,
    db: Session = Depends(get_db),
    page: int = 1,
    limit: int = 50
):
    offset = (page - 1) * limit
    advances = db.query(
        User.username,
        Work.work_date,
        Work.advance_amount
    ).join(User).filter(
        Work.advance_amount > 0
    ).order_by(
        Work.work_date.desc()
    ).offset(offset).limit(limit).all()

    total = db.query(Work).filter(Work.advance_amount > 0).count()

    return templates.TemplateResponse("advances.html", {
        "request": request,
        "advances": advances,
        "current_page": page,
        "total_pages": (total + limit - 1) // limit,
        "limit": limit
    })

@app.get("/generate-pdf")
async def generate_pdf(
    request: Request,
    db: Session = Depends(get_db),
    year: int = None,
    month: int = None,
    price_per_kg: float = 48.214
):
    # Get the same data as salary report
    salary_report_data = await salary_report(request, db, year, month, price_per_kg)
    context = salary_report_data.context
    
    # Add current datetime to context
    context["now"] = datetime.now()
    
    # Render PDF template
    pdf_html = templates.get_template("salary_pdf.html").render(context)

    
    # Create PDF
    pdf = io.BytesIO()
    pisa.CreatePDF(pdf_html, dest=pdf)
    pdf.seek(0)
    
    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=salary_report_{year}_{month}.pdf"}
    )

@app.get("/daily-tea")
async def daily_tea_report(
    request: Request,
    db: Session = Depends(get_db),
    year: int = None,
    month: int = None
):
    # Default to current month/year if not provided
    now = datetime.now()
    selected_year = year or now.year
    selected_month = month or now.month

    # Get tea weights from works
    works_data = db.query(
        Work.work_date,
        func.sum(Work.adjusted_tea_weight).label('total_adjusted')
    ).filter(
        func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
        func.strftime('%m', Work.work_date) == f"{selected_month:02d}"
    ).group_by(Work.work_date).all()

    # Get all factories
    factories = db.query(Factory).all()
    factory_map = {f.id: f.name for f in factories}

    # Get factory tea data per factory
    factory_data = db.query(
        FactoryTea.factory_date,
        FactoryTea.factory_id,
        func.sum(case((FactoryTea.weight_type == 'verified', FactoryTea.leaves_weight), else_=0)).label('verified'),
        func.sum(case((FactoryTea.weight_type == 'unverified', FactoryTea.leaves_weight), else_=0)).label('unverified')
    ).filter(
        func.strftime('%Y', FactoryTea.factory_date) == f"{selected_year:04d}",
        func.strftime('%m', FactoryTea.factory_date) == f"{selected_month:02d}"
    ).group_by(FactoryTea.factory_date, FactoryTea.factory_id).all()

    # Process factory data
    factory_dict = defaultdict(dict)
    for date, factory_id, v, u in factory_data:
        factory_dict[date][factory_id] = {
            'verified': v,
            'unverified': u,
            'name': factory_map.get(factory_id, f"Factory {factory_id}")
        }

    # Combine data
    daily_data = []
    date_set = set()
    
    # Process works data
    works_dict = {date: total for date, total in works_data}
    date_set.update([date for date, _ in works_data])
    
    # Process factory data
    date_set.update(factory_dict.keys())
    
    # Process factory data into daily_data
    for date in sorted(date_set, reverse=True):
        entry = {
            "date": date,
            "adjusted_tea": works_dict.get(date, 0.0),
            "factories": {}
        }
        
        # Add factory data
        if date in factory_dict:
            for factory_id, values in factory_dict[date].items():
                entry["factories"][factory_id] = {
                    "verified": values['verified'],
                    "unverified": values['unverified']
                }
        
        daily_data.append(entry)

    # Calculate factory totals
    factory_totals = defaultdict(lambda: {'verified': 0.0, 'unverified': 0.0})
    for entry in daily_data:
        for factory_id, values in entry["factories"].items():
            factory_totals[factory_id]['verified'] += values['verified']
            factory_totals[factory_id]['unverified'] += values['unverified']

    return templates.TemplateResponse("daily_tea.html", {
        "request": request,
        "daily_data": daily_data,
        "selected_year": selected_year,
        "selected_month": selected_month,
        "years": list(range(2020, now.year + 1)),
        "total_adjusted": sum(d['adjusted_tea'] for d in daily_data),
        "total_verified": sum(d['factories'][factory_id]['verified'] for d in daily_data for factory_id in d['factories']),
        "total_unverified": sum(d['factories'][factory_id]['unverified'] for d in daily_data for factory_id in d['factories']),
        "factories": factories,
        "factory_totals": factory_totals
    })

# Add server runner
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
