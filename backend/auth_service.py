"""
Secure Authentication System with Role-Based Access Control
Supports Admin, Analyst, and Moderator roles with JWT tokens
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import sqlite3
import hashlib
import secrets
from enum import Enum
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# User roles enum
class UserRole(Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    MODERATOR = "moderator"

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: UserRole
    full_name: str
    department: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    full_name: str
    department: Optional[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse
    expires_in: int

class AuthService:
    def __init__(self, db_path: str = "auth.db"):
        self.db_path = db_path
        self.init_database()
        self.create_default_admin()
    
    def init_database(self):
        """Initialize authentication database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                full_name TEXT NOT NULL,
                department TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                token_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                permission TEXT NOT NULL,
                UNIQUE(role, permission)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.init_permissions()
    
    def init_permissions(self):
        """Initialize role-based permissions"""
        permissions = {
            UserRole.ADMIN.value: [
                "dashboard.view", "dashboard.export", "users.create", "users.edit", 
                "users.delete", "users.view", "comments.view", "comments.moderate",
                "comments.delete", "reports.generate", "reports.export", "system.configure"
            ],
            UserRole.ANALYST.value: [
                "dashboard.view", "dashboard.export", "comments.view", 
                "reports.generate", "reports.export"
            ],
            UserRole.MODERATOR.value: [
                "dashboard.view", "comments.view", "comments.moderate", 
                "comments.edit", "reports.view"
            ]
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for role, perms in permissions.items():
            for perm in perms:
                cursor.execute(
                    "INSERT OR IGNORE INTO user_permissions (role, permission) VALUES (?, ?)",
                    (role, perm)
                )
        
        conn.commit()
        conn.close()
    
    def create_default_admin(self):
        """Create default admin user if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE role = ?", (UserRole.ADMIN.value,))
        admin_count = cursor.fetchone()[0]
        
        if admin_count == 0:
            admin_password = "Admin@123"
            password_hash = pwd_context.hash(admin_password)
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role, full_name, department)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                "admin", "admin@gov.in", password_hash, UserRole.ADMIN.value,
                "System Administrator", "IT Department"
            ))
            
            conn.commit()
            logger.info("Default admin user created - Username: admin, Password: Admin@123")
        
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        """Authenticate user with username and password"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if user is locked
        cursor.execute('''
            SELECT * FROM users 
            WHERE username = ? AND is_active = TRUE
        ''', (username,))
        
        user = cursor.fetchone()
        if not user:
            conn.close()
            return None
        
        user_id, username_db, email, password_hash, role, full_name, dept, is_active, created_at, last_login, failed_attempts, locked_until = user
        
        # Check if user is locked
        if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is temporarily locked due to multiple failed attempts"
            )
        
        # Verify password
        if not self.verify_password(password, password_hash):
            # Increment failed attempts
            cursor.execute('''
                UPDATE users SET failed_attempts = failed_attempts + 1,
                locked_until = CASE 
                    WHEN failed_attempts >= 4 THEN datetime('now', '+30 minutes')
                    ELSE locked_until
                END
                WHERE id = ?
            ''', (user_id,))
            conn.commit()
            conn.close()
            return None
        
        # Reset failed attempts and update last login
        cursor.execute('''
            UPDATE users SET 
                failed_attempts = 0, 
                locked_until = NULL, 
                last_login = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
        
        return {
            "id": user_id,
            "username": username_db,
            "email": email,
            "role": role,
            "full_name": full_name,
            "department": dept,
            "is_active": is_active,
            "created_at": created_at,
            "last_login": datetime.now().isoformat()
        }
    
    def get_user_by_username(self, username: str) -> Optional[dict]:
        """Get user by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, role, full_name, department, is_active, created_at, last_login
            FROM users WHERE username = ? AND is_active = TRUE
        ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                "id": user[0],
                "username": user[1],
                "email": user[2],
                "role": user[3],
                "full_name": user[4],
                "department": user[5],
                "is_active": user[6],
                "created_at": user[7],
                "last_login": user[8]
            }
        return None
    
    def create_user(self, user_data: UserCreate) -> dict:
        """Create new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if username or email already exists
        cursor.execute('''
            SELECT COUNT(*) FROM users WHERE username = ? OR email = ?
        ''', (user_data.username, user_data.email))
        
        if cursor.fetchone()[0] > 0:
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already exists"
            )
        
        password_hash = self.hash_password(user_data.password)
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, role, full_name, department)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_data.username, user_data.email, password_hash,
            user_data.role.value, user_data.full_name, user_data.department
        ))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return self.get_user_by_username(user_data.username)
    
    def has_permission(self, role: str, permission: str) -> bool:
        """Check if role has specific permission"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM user_permissions 
            WHERE role = ? AND permission = ?
        ''', (role, permission))
        
        has_perm = cursor.fetchone()[0] > 0
        conn.close()
        return has_perm

# Initialize auth service
auth_service = AuthService()

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    token = credentials.credentials
    payload = auth_service.verify_token(token)
    username = payload.get("sub")
    
    user = auth_service.get_user_by_username(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user

# Permission-based dependencies
def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_checker(current_user: dict = Depends(get_current_user)):
        if not auth_service.has_permission(current_user["role"], permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        return current_user
    return permission_checker

# Role-based dependencies
def require_admin(current_user: dict = Depends(get_current_user)):
    """Require admin role"""
    if current_user["role"] != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

def require_analyst_or_admin(current_user: dict = Depends(get_current_user)):
    """Require analyst or admin role"""
    if current_user["role"] not in [UserRole.ANALYST.value, UserRole.ADMIN.value]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analyst or Admin access required"
        )
    return current_user

def require_moderator_or_admin(current_user: dict = Depends(get_current_user)):
    """Require moderator or admin role"""
    if current_user["role"] not in [UserRole.MODERATOR.value, UserRole.ADMIN.value]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Moderator or Admin access required"
        )
    return current_user

# FastAPI app for authentication
auth_app = FastAPI(title="Government Policy Feedback - Authentication Service")

auth_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@auth_app.post("/auth/login", response_model=Token)
async def login(user_login: UserLogin):
    """User login endpoint"""
    user = auth_service.authenticate_user(user_login.username, user_login.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_service.create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@auth_app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, current_user: dict = Depends(require_admin)):
    """Register new user (Admin only)"""
    return auth_service.create_user(user_data)

@auth_app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@auth_app.post("/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """User logout endpoint"""
    # In a production system, you would invalidate the token
    return {"message": "Successfully logged out"}

@auth_app.get("/auth/permissions")
async def get_user_permissions(current_user: dict = Depends(get_current_user)):
    """Get current user's permissions"""
    conn = sqlite3.connect(auth_service.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT permission FROM user_permissions WHERE role = ?
    ''', (current_user["role"],))
    
    permissions = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return {"permissions": permissions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(auth_app, host="0.0.0.0", port=8001)