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
from datetime import datetime, date

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
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
    created_at = Column(DateTime, server_default=func.now())
    work_date = Column(Date, nullable=False, server_default=func.current_date())

    def __repr__(self):
        return f"<Work(id={self.id}, user_id={self.user_id}, date={self.work_date})>"

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

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
        "limit": limit
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

    # Create new work record
    work = Work(
        user_id=user_id,
        tea_weight=tea_weight_float,
        tea_location=tea_location,
        other_cost=other_cost_float,
        other_location=other_location,
        advance_amount=advance_amount_float,
        work_date=work_date_obj
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
        func.sum(Work.tea_weight).label('total_tea')
    ).join(User).filter(
        Work.tea_weight > 0
    ).group_by(
        Work.user_id,
        Work.work_date,
        Work.tea_location
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
        func.sum(Work.other_cost).label('total_cost')
    ).join(User).filter(
        Work.other_cost > 0
    ).group_by(
        Work.user_id,
        Work.work_date,
        Work.other_location
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
        func.sum(Work.tea_weight)
    ).filter(
        func.strftime('%Y', Work.work_date) == f"{selected_year:04d}",
        func.strftime('%m', Work.work_date) == f"{selected_month:02d}",
        Work.tea_weight > 0
    ).scalar() or 0.0

    # Generate year options (2020 to current year)
    current_year = now.year
    years = list(range(2020, current_year + 1))
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "monthly_total": monthly_total,
        "years": years,
        "selected_year": selected_year,
        "selected_month": selected_month
    })

# Add server runner
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
