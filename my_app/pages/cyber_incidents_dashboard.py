"""
Cyber Incidents Dashboard (DB + CSV)
 - Loads cyber_incidents from the SQLite DB (DATA/intelligence_platform.db)
 - via sqlite3 + pandas.
 - Loads DATA/cyber_incidents.csv directly.
 - Lets the user choose: view DB table / CSV / combined dataset.
 - Displays KPIs, charts, and the table. Allows adding a new incident (via DB insert function if available).
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


#set_page_config before anything that writes to the page
st.set_page_config(page_title="Cyber Incidents (DB + CSV)", layout="wide", page_icon="🛡️")

# Authentication guard
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.error("You must be logged in to view the dashboard.")
    if st.button("Go to login"):
        st.switch_page("Home.py")
    st.stop()

# Paths
DB_PATH = Path("DATA") / "intelligence_platform.db"
CSV_PATH = Path("DATA") / "cyber_incidents.csv"


def connect_via_helper(db_path: Path):
    """Try to connect via app.data.db helper if available."""
    try:
        from app.data.db import connect_database  # type: ignore
        return connect_database(str(db_path))
    except Exception:
        raise

def connect_fallback(db_path: Path):
    """Fallback sqlite3 connection via pandas-friendly sqlite3."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return conn

@st.cache_resource
def get_connection(db_path: Path = DB_PATH) -> Optional[sqlite3.Connection]:
    """
    Return a cached DB connection resource or None if DB not present/openable.
    This is safe to cache_resource because it's a resource (not hashed by st.cache_data).
    """
    if not db_path.exists():
        st.warning(f"Database file not found at {db_path}. DB features will be disabled.")
        return None
    try:
        conn = connect_via_helper(db_path)
        return conn
    except Exception:
        # fallback to sqlite3
        try:
            conn = connect_fallback(db_path)
            return conn
        except Exception as e:
            st.error(f"Unable to open DB: {e}")
            return None
        


# Load DB table into DataFrame
# NOTE: Do NOT accept a Connection object as an argument to a cached function,
# because sqlite3.Connection is unhashable. Instead request the connection inside the function.
@st.cache_data(ttl=60)
def load_db_table(table_name: str = "cyber_incidents") -> pd.DataFrame:
    """
    Load the named table from the DB. The connection is obtained from get_connection()
    inside the function, avoiding unhashable parameters.
    """
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        # helpful fallback: try to discover available tables
        try:
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            st.info(f"DB tables: {tables['name'].tolist()}")
        except Exception:
            pass
        st.error(f"Failed to read table '{table_name}' from DB: {e}")
        return pd.DataFrame()

# Load CSV
@st.cache_data(ttl=60)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.warning(f"CSV not found at {path}.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV {path}: {e}")
        return pd.DataFrame()

# Try to import insert function to allow adding into DB if available
def try_get_insert_function():
    """
    If you've got app.data.incidents.insert_incident, return it.
    Otherwise return None.
    """
    try:
        from app.data.incidents import insert_incident  # type: ignore
        return insert_incident
    except Exception:
        return None

# Load data
# get_connection() is cached_resource; load_db_table() will call it internally
db_df = load_db_table()  # no conn argument anymore
csv_df = load_csv(CSV_PATH)

# Merge/choose options
st.title("🔐 Cyber Incidents")
st.subheader(f"Hello, {st.session_state.username} — choose source to view")

source = st.radio(
    "Data source",
    options=["Database table (DB)", "CSV file", "Combined (DB + CSV)"],
    index=0,
)

# Prepare combined dataset
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure consistent columns and types for easier combining:
    - Trim column names
    - Map common variations to canonical names
    - Parse date column if present
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    names = [c.lower() for c in df.columns]
    if "date" not in names:
        for alt in ("incident_date", "reported_on", "created_at"):
            if alt in names:
                orig = df.columns[names.index(alt)]
                colmap[orig] = "date"
                break
    if "severity" not in names:
        for alt in ("level", "impact"):
            if alt in names:
                orig = df.columns[names.index(alt)]
                colmap[orig] = "severity"
                break
    if "status" not in names:
        for alt in ("state", "incident_status"):
            if alt in names:
                orig = df.columns[names.index(alt)]
                colmap[orig] = "status"
                break
    if colmap:
        df = df.rename(columns=colmap)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            pass
    return df

db_df_norm = normalize_df(db_df)
csv_df_norm = normalize_df(csv_df)

if source == "Database table (DB)":
    df_display = db_df_norm.copy()
elif source == "CSV file":
    df_display = csv_df_norm.copy()
else:
    # Combined: concat and drop duplicates (use 'id' if available)
    combined = pd.concat([db_df_norm, csv_df_norm], ignore_index=True, sort=False)
    if "id" in combined.columns:
        combined = combined.drop_duplicates(subset=["id"])
    else:
        dedupe_keys = []
        if "title" in combined.columns:
            dedupe_keys.append("title")
        if "date" in combined.columns:
            dedupe_keys.append("date")
        if dedupe_keys:
            combined = combined.drop_duplicates(subset=dedupe_keys)
        else:
            combined = combined.drop_duplicates()
    df_display = combined.reset_index(drop=True)

# KPIs
total = len(df_display)
open_cnt = int((df_display.get("status") == "Open").sum()) if "status" in df_display.columns else 0
critical_cnt = int((df_display.get("severity") == "Critical").sum()) if "severity" in df_display.columns else 0

col1, col2, col3 = st.columns(3)
col1.metric("Displayed incidents", total)
col2.metric("Open", open_cnt)
col3.metric("Critical", critical_cnt)

st.divider()

# Sidebar filters (applies to displayed df)
with st.sidebar:
    st.header("Filters")
    if "date" in df_display.columns and not df_display["date"].isna().all():
        min_date = pd.to_datetime(df_display["date"]).min().date()
        max_date = pd.to_datetime(df_display["date"]).max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date))
    else:
        date_range = None

    if "severity" in df_display.columns:
        avail = sorted(df_display["severity"].dropna().unique().tolist())
        severity_sel = st.multiselect("Severity", options=avail, default=avail)
    else:
        severity_sel = []

    if "status" in df_display.columns:
        avail = sorted(df_display["status"].dropna().unique().tolist())
        status_sel = st.multiselect("Status", options=avail, default=avail)
    else:
        status_sel = []

    if st.button("Refresh data"):
        # Clear caches and reload
        load_db_table.clear()
        load_csv.clear()
        st.rerun()

# Apply filters
df_filtered = df_display.copy()
if date_range is not None and "date" in df_filtered.columns:
    start_date, end_date = date_range
    df_filtered = df_filtered[
        (pd.to_datetime(df_filtered["date"]).dt.date >= start_date)
        & (pd.to_datetime(df_filtered["date"]).dt.date <= end_date)
    ]

if severity_sel and "severity" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["severity"].isin(severity_sel)]

if status_sel and "status" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["status"].isin(status_sel)]

# Charts + table
st.subheader("Incidents Overview")

chart_col, table_col = st.columns((1, 2))

with chart_col:
    if "severity" in df_filtered.columns and not df_filtered["severity"].isna().all():
        severity_count = df_filtered["severity"].value_counts()
        st.bar_chart(severity_count)

    if "date" in df_filtered.columns and not df_filtered["date"].isna().all():
        ts = df_filtered.groupby(pd.to_datetime(df_filtered["date"]).dt.date).size().rename("count")
        st.line_chart(ts)

with table_col:
    st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True)

with st.expander("Show raw (unfiltered) dataset"):
    st.write(df_display)

st.divider()

 #Insert new incident 
insert_func = try_get_insert_function()
conn = get_connection()
can_insert = (conn is not None) and (insert_func is not None)

if can_insert:
    with st.form("add_incident_db"):
        t = st.text_input("Title")
        sev = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"])
        status = st.selectbox("Status", ["Open", "In Progress", "Resolved"])
        date_val = st.date_input("Date", value=datetime.today().date())

        submitted = st.form_submit_button("Add to DB")

        if submitted:
            if not t.strip():
                st.error("Title required.")
            else:
                try:
                    # Expected signature:
                    # insert_incident(conn, title, severity, status, date=...)
                    insert_func(conn, t.strip(), sev, status, date=date_val)
                    st.success("Incident successfully added to the database.")
                    load_db_table.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Insert failed: {e}")
else:
    st.info(
        "Database insert is unavailable. "
        "Ensure the database exists and `insert_incident` is implemented."
    )

# Logout button
st.divider()

if st.button("Log out"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out")
    st.switch_page("Home.py")