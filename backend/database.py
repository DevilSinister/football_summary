import os
import re
from urllib.parse import quote_plus

import pyodbc
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


SQL_SERVER = os.getenv("FOOTY_SQL_SERVER", r".\SQLEXPRESS")
SQL_DATABASE = os.getenv("FOOTY_SQL_DATABASE", "FootballDB")
ODBC_DRIVER = os.getenv("FOOTY_ODBC_DRIVER", "ODBC Driver 18 for SQL Server")


def _odbc_connection_string(database: str) -> str:
    return (
        f"DRIVER={{{ODBC_DRIVER}}};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;Encrypt=yes;TrustServerCertificate=yes;"
    )


def ensure_database_exists() -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]+", SQL_DATABASE):
        raise ValueError("FOOTY_SQL_DATABASE may contain only letters, numbers, and underscores")
    connection = pyodbc.connect(_odbc_connection_string("master"), autocommit=True)
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DB_ID(?)", SQL_DATABASE)
        if cursor.fetchone()[0] is None:
            cursor.execute(f"CREATE DATABASE [{SQL_DATABASE}]")
    finally:
        connection.close()


ensure_database_exists()
SQLALCHEMY_DATABASE_URL = (
    "mssql+pyodbc:///?odbc_connect=" + quote_plus(_odbc_connection_string(SQL_DATABASE))
)
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db():
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()
