from fastapi import FastAPI, Request, Depends, Form, Cookie
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, ForeignKey, Date, case, Enum as SQLAlchemyEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import uvicorn
from typing import Optional
from fastapi import HTTPException
from datetime import datetime, date, timedelta
from sqlalchemy.exc import IntegrityError
from calendar import monthcalendar
from fastapi.responses import StreamingResponse
import io
from collections import defaultdict
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from enum import Enum
import pyppeteer
import asyncio
import os

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
Base = declarative_base()

class UserRole(str, Enum):
    ADMIN = 'admin'
    USER = 'user'

# INSERT INTO admin (username, email, hashed_password, role)VALUES (    'admin_root',    'root@root.com',    '$2b$12$MUXDZvtmHAWdFfsaVgmide3//3RnCzp24ssnvOQ6MI3Yw9LPzq0Qi',    'ADMIN')ON CONFLICT(email) DO NOTHING;

class AdminUser(Base):
    __tablename__ = "admin"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    email = Column(String(100), unique=True)
    hashed_password = Column(String(100))
    role = Column(SQLAlchemyEnum(UserRole, name='userrole_enum'), default=UserRole.USER)

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

# Add this line to mount static files (should be after app creation and before routes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Add custom Jinja filters and globals
def format_commas(value):
    return "{:,.0f}".format(float(value)) if float(value) % 1 == 0 else "{:,.2f}".format(float(value))

templates.env.filters["thousands_commas"] = format_commas
templates.env.globals.update({
    "min": min,  # Add the min function to template globals
    "max": max   # Also add max since we're using it in the template
})

# Replace Flask-Login initialization with
SECRET_KEY = "your-secret-key-keep-it-secret"
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Add authentication utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_current_user(
    request: Request,
    db: Session = Depends(get_db)
):
    # Get token from cookies first
    token = request.cookies.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Clean token format
    if token.startswith("Bearer "):
        token = token[7:]

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        if not user_email:
            raise HTTPException(status_code=401, detail="Invalid token format")
        
        user = db.query(AdminUser).filter(AdminUser.email == user_email).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        # Convert Enum to string for templates
        user.role = user.role.value
        return user
        
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

@app.get("/")
async def read_root():
    return RedirectResponse(url="/dashboard")

# Add ping endpoint
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/users")
async def users_page(request: Request, db: Session = Depends(get_db), 
    current_user: AdminUser = Depends(get_current_user)):
    users = db.query(User).all()
    return templates.TemplateResponse("users.html", {"request": request, "users": users, "current_user": current_user})

@app.post("/users")
async def create_user(
    request: Request,
    username: str = Form(...),
    paytype: list[str] = Form(...),
    basic_salary: str = Form(""),
    weekly_salary: str = Form(""),
    db: Session = Depends(get_db), 
    current_user: AdminUser = Depends(get_current_user)
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
    limit: int = 10, 
    current_user: AdminUser = Depends(get_current_user)
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
        "factories": db.query(Factory).all(), 
        "current_user": current_user
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
    db: Session = Depends(get_db),
    current_user: AdminUser = Depends(get_current_user)
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
async def tea_leaves_report(
    request: Request,
    page: int = 1,
    user_id: Optional[str] = None,
    from_date: Optional[str] = None,  # Add date filter
    db: Session = Depends(get_db),
    current_user: AdminUser = Depends(get_current_user)
):
    per_page = 10
    
    # Base query
    query = db.query(
        Work.work_date,
        User.username,
        Work.tea_location,
        Work.tea_weight,
        Work.adjusted_tea_weight
    ).join(User).filter(Work.tea_weight > 0)
    
    # Apply filters
    if user_id and user_id != '':
        query = query.filter(Work.user_id == int(user_id))
    if from_date:
        try:
            date_obj = datetime.strptime(from_date, '%Y-%m-%d').date()
            query = query.filter(Work.work_date >= date_obj)
        except ValueError:
            pass  # Invalid date format, ignore filter
    
    # Get total count
    total_records = query.count()
    total_pages = (total_records + per_page - 1) // per_page
    
    # Get paginated records
    records = query.order_by(
        Work.work_date.desc()
    ).offset((page - 1) * per_page).limit(per_page).all()
    
    # Get all users for filter dropdown
    users = db.query(User).order_by(User.username).all()

    return templates.TemplateResponse("tea_leaves.html", {
        "request": request,
        "records": records,
        "current_user": current_user,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total_records": total_records,
        "users": users,
        "selected_user_id": user_id,
        "from_date": from_date
    })

@app.get("/other-work")
async def other_work_report(
    request: Request,
    page: int = 1,
    user_id: Optional[str] = None,
    from_date: Optional[str] = None,  # Add date filter
    db: Session = Depends(get_db),
    current_user: AdminUser = Depends(get_current_user)
):
    per_page = 10
    
    # Base query
    query = db.query(
        Work.work_date,
        User.username,
        Work.other_location,
        Work.work_description,
        func.sum(Work.other_cost).label('total_cost')
    ).join(User).filter(Work.other_cost > 0)
    
    # Apply filters
    if user_id and user_id != '':
        query = query.filter(Work.user_id == int(user_id))
    if from_date:
        try:
            date_obj = datetime.strptime(from_date, '%Y-%m-%d').date()
            query = query.filter(Work.work_date >= date_obj)
        except ValueError:
            pass  # Invalid date format, ignore filter
    
    # Get total count (need to count before grouping)
    count_query = db.query(Work).join(User).filter(Work.other_cost > 0)
    if user_id:
        count_query = count_query.filter(Work.user_id == user_id)
    if from_date:
        try:
            date_obj = datetime.strptime(from_date, '%Y-%m-%d').date()
            count_query = count_query.filter(Work.work_date >= date_obj)
        except ValueError:
            pass
    
    total_records = count_query.count()
    total_pages = (total_records + per_page - 1) // per_page
    
    # Complete the query with group by and pagination
    results = query.group_by(
        Work.user_id,
        Work.work_date,
        Work.other_location,
        Work.work_description
    ).order_by(
        Work.work_date.desc()
    ).offset((page - 1) * per_page).limit(per_page).all()
    
    # Get all users for filter dropdown
    users = db.query(User).order_by(User.username).all()

    return templates.TemplateResponse("other_work.html", {
        "request": request,
        "records": results,
        "current_user": current_user,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total_records": total_records,
        "users": users,
        "selected_user_id": user_id,
        "from_date": from_date
    })

@app.get("/dashboard")
async def dashboard(
    request: Request, 
    current_user: AdminUser = Depends(get_current_user),
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

    # Get location-based totals
    location_totals = db.query(
        Work.tea_location,
        func.sum(Work.adjusted_tea_weight).label('total_weight')
    ).filter(
        func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
        func.strftime('%m', Work.work_date) == f"{selected_month:02d}",
        Work.adjusted_tea_weight > 0
    ).group_by(Work.tea_location).all()

    location_labels = [loc.tea_location or 'Unknown' for loc in location_totals]
    location_data = [float(loc.total_weight) for loc in location_totals]

    # Get work cost totals by location
    work_cost_totals = db.query(
        Work.other_location,
        func.sum(Work.other_cost).label('total_cost')
    ).filter(
        func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
        func.strftime('%m', Work.work_date) == f"{selected_month:02d}",
        Work.other_cost > 0
    ).group_by(Work.other_location).all()

    # Prepare work cost data for the chart
    work_cost_data = [float(loc.total_cost) for loc in work_cost_totals]
    location_labels_work = [loc.other_location or 'Unknown' for loc in work_cost_totals]

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
        "location_datasets": location_datasets,
        "location_labels": location_labels,
        "location_labels_work": location_labels_work,
        "location_data": location_data,
        "work_cost_data": work_cost_data,
        "current_user": current_user
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
    price_per_kg: float = 48.214, 
    current_user: AdminUser = Depends(get_current_user)
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

        # Count extra work days
        aththam_days = db.query(
            func.count(func.distinct(Work.work_date))
        ).filter(
            Work.user_id == user.id,
            Work.work_description.startswith('Aththa'),
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
            "extra_days": extra_days,
            "aththam_days": aththam_days
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
        "weekly_salary_data": weekly_salary_data,
        "current_user": current_user
    }

    return templates.TemplateResponse("salary.html", context)

@app.get("/factories")
async def factories_page(request: Request, db: Session = Depends(get_db), 
    current_user: AdminUser = Depends(get_current_user)):
    factories = db.query(Factory).order_by(Factory.created_at.desc()).all()
    return templates.TemplateResponse("factories.html", {
        "request": request,
        "factories": factories,
        "current_user": current_user
    })

@app.post("/factories")
async def create_factory(
    request: Request,
    name: str = Form(...),
    db: Session = Depends(get_db), 
    current_user: AdminUser = Depends(get_current_user)
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
    db: Session = Depends(get_db), 
    current_user: AdminUser = Depends(get_current_user)
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
    page: int = 1,
    user_id: Optional[str] = None,
    from_date: Optional[str] = None,  # Add date filter
    db: Session = Depends(get_db),
    current_user: AdminUser = Depends(get_current_user)
):
    per_page = 10
    
    # Base query
    query = db.query(
        User.username,
        Work.work_date,
        Work.advance_amount
    ).join(User).filter(Work.advance_amount > 0)
    
    # Apply filters
    if user_id and user_id != '':
        query = query.filter(Work.user_id == int(user_id))
    if from_date:
        try:
            date_obj = datetime.strptime(from_date, '%Y-%m-%d').date()
            query = query.filter(Work.work_date >= date_obj)
        except ValueError:
            pass  # Invalid date format, ignore filter
    
    # Get total count
    total_records = query.count()
    total_pages = (total_records + per_page - 1) // per_page
    
    # Get paginated records
    advances = query.order_by(
        Work.work_date.desc()
    ).offset((page - 1) * per_page).limit(per_page).all()
    
    # Get all users for filter dropdown
    users = db.query(User).order_by(User.username).all()

    return templates.TemplateResponse("advances.html", {
        "request": request,
        "advances": advances,
        "current_user": current_user,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total_records": total_records,
        "users": users,
        "selected_user_id": user_id,
        "from_date": from_date
    })

@app.get("/generate-pdf")
async def generate_pdf(
    request: Request,
    db: Session = Depends(get_db),
    year: int = None,
    month: int = None,
    price_per_kg: float = 48.214, 
    current_user: AdminUser = Depends(get_current_user)
):
    browser = None
    temp_html_path = None
    temp_pdf_path = None

    try:
        # Get the same data as salary report
        salary_report_data = await salary_report(request, db, year, month, price_per_kg)
        context = salary_report_data.context
        
        # Add current datetime to context
        context["now"] = datetime.now()
        
        # Render HTML template
        html_content = templates.get_template("salary_pdf_sinhala.html").render(context)
        
        # Create a temporary file for the HTML content
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_html_path = os.path.join(temp_dir, "temp_salary.html")
        temp_pdf_path = os.path.join(temp_dir, "temp_salary.pdf")
        
        with open(temp_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Launch browser with more robust options
        browser = await pyppeteer.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--disable-gpu',
                '--font-render-hinting=none'
            ],
            handleSIGINT=False,
            handleSIGTERM=False,
            handleSIGHUP=False,
            ignoreHTTPSErrors=True
        )

        # Create a new page with a longer timeout
        page = await browser.newPage()
        
        # Set viewport and load HTML file
        await page.setViewport({'width': 1024, 'height': 768})
        
        # Load the HTML file with a longer timeout
        await page.goto(
            f'file://{os.path.abspath(temp_html_path)}',
            waitUntil=['networkidle0', 'load'],
            timeout=300000
        )

        # Wait for fonts to load and content to render
        await page.evaluate('() => new Promise(resolve => setTimeout(resolve, 20000))')

        # Generate PDF with more specific options
        await page.pdf({
            'path': temp_pdf_path,
            'format': 'A4',
            'printBackground': True,
            'margin': {'top': '20mm', 'right': '20mm', 'bottom': '20mm', 'left': '20mm'},
            'preferCSSPageSize': True
        })

        # Read the generated PDF
        with open(temp_pdf_path, "rb") as f:
            pdf_content = f.read()

        # Return the PDF as a streaming response
        return StreamingResponse(
            io.BytesIO(pdf_content),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=salary_report_{year}_{month}.pdf"}
        )

    except Exception as e:
        error_message = f"PDF generation failed: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

    finally:
        # Clean up resources in finally block
        if browser:
            try:
                await browser.close()
            except Exception as e:
                print(f"Browser cleanup error: {str(e)}")

        # Clean up temporary files
        try:
            if temp_html_path and os.path.exists(temp_html_path):
                os.remove(temp_html_path)
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
        except Exception as e:
            print(f"File cleanup error: {str(e)}")

@app.get("/daily-tea")
async def daily_tea_report(
    request: Request,
    db: Session = Depends(get_db),
    year: int = None,
    month: int = None,
    current_user: AdminUser = Depends(get_current_user)
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
        "total_adjusted": sum(d['adjusted_tea'] if d['adjusted_tea'] is not None else 0 for d in daily_data),
        "total_verified": sum(d['factories'][factory_id]['verified'] for d in daily_data for factory_id in d['factories']),
        "total_unverified": sum(d['factories'][factory_id]['unverified'] for d in daily_data for factory_id in d['factories']),
        "factories": factories,
        "factory_totals": factory_totals,
        "current_user": current_user
    })

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(..., alias="email"),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(AdminUser).filter(AdminUser.email == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid credentials"
        })
    
    access_token = jwt.encode(
        {"sub": user.email}, SECRET_KEY, algorithm=ALGORITHM
    )
    
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        key="Authorization",
        value=f"Bearer {access_token}",
        httponly=True,
        max_age=3600,  # 1 hour
        secure=False,  # Allow in HTTP for development
        samesite="Lax"  # Enable cookie sending across same-site
    )
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("Authorization")
    return response

@app.post("/token")
async def token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(AdminUser).filter(AdminUser.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
        )
    
    access_token = jwt.encode(
        {"sub": user.email}, SECRET_KEY, algorithm=ALGORITHM
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Allow login page without token
    print(request.url.path)
    if request.url.path == "/login":
        return await call_next(request)
    
    # Existing token check
    token = request.cookies.get("Authorization")
    if not token:
        return RedirectResponse(url="/login")
    
    response = await call_next(request)
    return response

@app.get("/revenue")
async def revenue_report(
    request: Request,
    db: Session = Depends(get_db),
    year: int = None,
    month: int = None,
    current_user: AdminUser = Depends(get_current_user)
):
    # Get selected month/year
    now = datetime.now()
    selected_year = year or now.year
    selected_month = month or now.month

    # Get factory tea data
    factories = db.query(Factory).all()
    factory_data = []
    
    # Get total tea leaves per factory
    for factory in factories:
        tea_data = db.query(
            func.sum(FactoryTea.leaves_weight).filter(FactoryTea.weight_type == 'verified').label('verified'),
            func.sum(FactoryTea.leaves_weight).filter(FactoryTea.weight_type == 'unverified').label('unverified')
        ).filter(
            FactoryTea.factory_id == factory.id,
            func.strftime('%Y', FactoryTea.factory_date) == f"{selected_year:04d}",
            func.strftime('%m', FactoryTea.factory_date) == f"{selected_month:02d}"
        ).first()
        
        factory_data.append({
            "id": factory.id,
            "name": factory.name,
            "verified": tea_data.verified or 0.0,
            "unverified": tea_data.unverified or 0.0
        })

    # Calculate total basic salaries for monthly paid users
    total_basic_salary = db.query(func.sum(User.basic_salary)).filter(
        User.paytype.like('%monthly%')
    ).scalar() or 0.0

    # Calculate total tea income (adjusted tea weight * standard price)
    price_per_kg = 48.214  # Standard tea price
    total_tea_income = (db.query(func.sum(Work.adjusted_tea_weight)).filter(
        func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
        func.strftime('%m', Work.work_date) == f"{selected_month:02d}"
    ).scalar() or 0.0) * price_per_kg

    total_payroll = total_basic_salary

    # Get other work expenses
    other_work_total = db.query(func.sum(Work.other_cost)).filter(
        func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
        func.strftime('%m', Work.work_date) == f"{selected_month:02d}"
    ).scalar() or 0.0

     # Process factory prices
    total_revenue = 0.0
    factory_revenues = []
    
    form_data = await request.form()

    for factory in factory_data:
        price_key = f"price_{factory['id']}"
        price = float(form_data.get(price_key, 200.0))  # Default to 200 if not provided
        
        total_tea = factory["verified"] + factory["unverified"]
        revenue = total_tea * price
        
        factory_revenues.append({
            **factory,
            "price": price,
            "revenue": revenue
        })
        total_revenue += revenue

    return templates.TemplateResponse("revenue.html", {
        "request": request,
        "factory_data": factory_data,
        "factory_revenues": factory_revenues,  # Initialize empty list
        "selected_year": selected_year,
        "selected_month": selected_month,
        "years": list(range(2020, now.year + 1)),
        "total_payroll": total_payroll,
        "total_tea_income": total_tea_income,
        "other_work_total": other_work_total,
        "total_revenue": total_revenue,  # Default value
        "total_profit": total_revenue - total_payroll - total_tea_income - other_work_total,   # Default value
        "current_user": current_user
    })

@app.post("/revenue")
async def calculate_revenue(
    request: Request,
    db: Session = Depends(get_db),
    year: int = None,
    month: int = None,
    current_user: AdminUser = Depends(get_current_user)
):
    # Get form data
    form_data = await request.form()
    
    # Get existing report data
    report_data = await revenue_report(request, db, year, month)
    context = report_data.context
    
    # Process factory prices
    total_revenue = 0.0
    factory_revenues = []
    
    for factory in context["factory_data"]:
        price_key = f"price_{factory['id']}"
        price = float(form_data.get(price_key, 200.0))  # Default to 200 if not provided
        
        total_tea = factory["verified"] + factory["unverified"]
        revenue = total_tea * price
        
        factory_revenues.append({
            **factory,
            "price": price,
            "revenue": revenue
        })
        total_revenue += revenue

    # Update context with calculations
    context.update({
        "factory_revenues": factory_revenues,
        "total_revenue": total_revenue,
        "total_profit": total_revenue - context["total_payroll"] - context["total_tea_income"] - context["other_work_total"]
    })

    return templates.TemplateResponse("revenue.html", context)

# Add server runner
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
