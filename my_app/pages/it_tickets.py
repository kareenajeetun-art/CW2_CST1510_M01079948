"""
IT Tickets Dashboard (DB + CSV)
 - Loads it_tickets from DATA/intelligence_platform.db (if present)
 - Loads DATA/it_tickets.csv (if present)
 - Choose DB / CSV / Combined, show KPIs, filters, charts, table.
 - Insert new ticket (calls app.data.it_tickets.insert_it_ticket if present).
 - Edit ticket (calls app.data.it_tickets.update_it_ticket if present).
 - Optional CSV -> DB loader using load_csv_to_table_it_tickets if present.
 - Uses Altair for nicer charts (falls back to Streamlit charts if Altair not available).
"""

import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
import inspect


try:
    import altair as alt  # type: ignore
    HAS_ALTAIR = True
except Exception:
    HAS_ALTAIR = False

# Page config (set before any writes)
st.set_page_config(page_title="IT Tickets (DB + CSV)", layout="wide", page_icon="🧰")

# Authentication guard (same as Home.py)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.error("You must be logged in to view the IT Tickets dashboard.")
    if st.button("Go to login"):
        st.switch_page("Home.py")
    st.stop()

# Paths / constants
DB_PATH = Path("DATA") / "intelligence_platform.db"
CSV_PATH = Path("DATA") / "it_tickets.csv"
TABLE_NAME = "it_tickets"

# DB connection helpers (try your app.data.db helper first)
def connect_via_helper(db_path: Path):
    try:
        from app.data.db import connect_database  # type: ignore
        try:
            return connect_database(str(db_path))
        except TypeError:
            return connect_database()
    except Exception:
        raise

def connect_fallback(db_path: Path):
    return sqlite3.connect(str(db_path), check_same_thread=False)

@st.cache_resource
def get_connection(db_path: Path = DB_PATH) -> Optional[sqlite3.Connection]:
    """Return cached DB connection or None if DB missing/unopenable."""
    if not db_path.exists():
        st.warning(f"Database file not found at {db_path}. DB features will be disabled.")
        return None
    try:
        return connect_via_helper(db_path)
    except Exception:
        try:
            return connect_fallback(db_path)
        except Exception as e:
            st.error(f"Unable to open DB: {e}")
            return None

# Load functions (DB table + CSV)
@st.cache_data(ttl=60)
def load_db_table(table_name: str = TABLE_NAME) -> pd.DataFrame:
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        q = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(q, conn)
        return df
    except Exception as e:
        try:
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            st.info(f"DB tables: {tables['name'].tolist()}")
        except Exception:
            pass
        st.error(f"Failed to read table '{table_name}' from DB: {e}")
        return pd.DataFrame()

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

# Try to import app.data.it_tickets functions if present
def try_get_insert_function() -> Optional[Callable]:
    try:
        from app.data.it_tickets import insert_it_ticket  # type: ignore
        return insert_it_ticket
    except Exception:
        return None

def try_get_update_function() -> Optional[Callable]:
    try:
        from app.data.it_tickets import update_it_ticket  # type: ignore
        return update_it_ticket
    except Exception:
        return None

def try_get_csv_loader() -> Optional[Callable]:
    try:
        from app.data.it_tickets import load_csv_to_table_it_tickets  # type: ignore
        return load_csv_to_table_it_tickets
    except Exception:
        return None

# Normalization helper for safe combining
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    names = [c.lower() for c in df.columns]
    colmap = {}

    # canonical mappings commonly used in your CSV/DB
    if "ticket_id" not in names:
        for alt in ("id", "ticket", "ticketid"):
            if alt in names:
                colmap[df.columns[names.index(alt)]] = "ticket_id"
                break

    # map obvious fields if differently named
    for alt in ("created_at", "created"):
        if alt in names and "created_date" not in names:
            colmap[df.columns[names.index(alt)]] = "created_date"
            break

    for alt in ("resolved_at", "resolved"):
        if alt in names and "resolved_date" not in names:
            colmap[df.columns[names.index(alt)]] = "resolved_date"
            break

    # others (priority, status, subject, description, assigned_to, category)
    for alt in ("priority", "status", "subject", "description", "assigned_to", "category"):
        if alt in names and alt not in names:
            # noop (keeps existing)
            pass

    if colmap:
        df = df.rename(columns=colmap)

    # coerce dates
    if "created_date" in df.columns:
        try:
            df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
        except Exception:
            pass
    if "resolved_date" in df.columns:
        try:
            df["resolved_date"] = pd.to_datetime(df["resolved_date"], errors="coerce")
        except Exception:
            pass

    return df

# Load dataframes
db_df = load_db_table()
csv_df = load_csv(CSV_PATH)

db_df_norm = normalize_df(db_df)
csv_df_norm = normalize_df(csv_df)

# UI: choose source 
st.title("🧰 IT Tickets")
st.subheader(f"Hello, {st.session_state.username} — choose source to view")

source = st.radio(
    "Data source",
    options=["Database table (DB)", "CSV file", "Combined (DB + CSV)"],
    index=0
)

if source == "Database table (DB)":
    df_display = db_df_norm.copy()
elif source == "CSV file":
    df_display = csv_df_norm.copy()
else:
    combined = pd.concat([db_df_norm, csv_df_norm], ignore_index=True, sort=False)
    # dedupe by id or subject+created_date
    if "id" in combined.columns:
        combined = combined.drop_duplicates(subset=["id"])
    elif "ticket_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["ticket_id"])
    else:
        keys = []
        if "subject" in combined.columns:
            keys.append("subject")
        if "created_date" in combined.columns:
            keys.append("created_date")
        if keys:
            combined = combined.drop_duplicates(subset=keys)
        else:
            combined = combined.drop_duplicates()
    df_display = combined.reset_index(drop=True)

# KPIs 
total = len(df_display)
open_cnt = int((df_display.get("status") == "Open").sum()) if "status" in df_display.columns else 0
high_priority = 0
if "priority" in df_display.columns:
    high_priority = int(df_display[df_display["priority"].isin(["High", "Critical", "P1", "P0"])].shape[0])

avg_resolution = None
if "created_date" in df_display.columns and "resolved_date" in df_display.columns:
    resolved_mask = df_display["resolved_date"].notna() & df_display["created_date"].notna()
    if resolved_mask.any():
        diffs = (pd.to_datetime(df_display.loc[resolved_mask, "resolved_date"]) - pd.to_datetime(df_display.loc[resolved_mask, "created_date"]))
        avg_hours = diffs.dt.total_seconds().mean() / 3600.0
        avg_resolution = round(avg_hours, 1)

col1, col2, col3 = st.columns(3)
col1.metric("Displayed tickets", total)
col2.metric("Open tickets", open_cnt)
col3.metric("Avg resolution (hrs)", avg_resolution if avg_resolution is not None else "N/A")

st.divider()

# Sidebar filters 
with st.sidebar:
    st.header("Filters")

    if "created_date" in df_display.columns and not df_display["created_date"].isna().all():
        min_date = pd.to_datetime(df_display["created_date"]).min().date()
        max_date = pd.to_datetime(df_display["created_date"]).max().date()
        date_range = st.date_input("Created date range", value=(min_date, max_date))
    else:
        date_range = None

    if "priority" in df_display.columns:
        pri_opts = sorted(df_display["priority"].dropna().unique().tolist())
        priority_sel = st.multiselect("Priority", options=pri_opts, default=pri_opts)
    else:
        priority_sel = []

    if "status" in df_display.columns:
        st_opts = sorted(df_display["status"].dropna().unique().tolist())
        status_sel = st.multiselect("Status", options=st_opts, default=st_opts)
    else:
        status_sel = []

    if "assigned_to" in df_display.columns:
        assignees = sorted(df_display["assigned_to"].dropna().unique().tolist())
        assigned_sel = st.multiselect("Assigned to", options=assignees, default=assignees)
    else:
        assigned_sel = []

    if st.button("Refresh data"):
        load_db_table.clear()
        load_csv.clear()
        st.rerun()

# Apply filters
df_filtered = df_display.copy()

if date_range is not None and "created_date" in df_filtered.columns:
    start_date, end_date = date_range
    df_filtered = df_filtered[
        (pd.to_datetime(df_filtered["created_date"]).dt.date >= start_date)
        & (pd.to_datetime(df_filtered["created_date"]).dt.date <= end_date)
    ]

if priority_sel and "priority" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["priority"].isin(priority_sel)]

if status_sel and "status" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["status"].isin(status_sel)]

if assigned_sel and "assigned_to" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["assigned_to"].isin(assigned_sel)]

# Charts + table (fancier) 
st.subheader("Tickets Overview")

chart_col, table_col = st.columns((1.2, 1.8))

with chart_col:
    # Stacked bar: Priority vs Status (if both present)
    if "priority" in df_filtered.columns and "status" in df_filtered.columns:
        st.markdown("**Priority × Status**")
        counts = (df_filtered.groupby(["priority", "status"])
                  .size()
                  .reset_index(name="count"))
        if HAS_ALTAIR and not counts.empty:
            chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X("priority:N", title="Priority", sort=alt.EncodingSortField(field="count", order="descending")),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color("status:N", title="Status"),
                tooltip=["priority", "status", "count"]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.bar_chart(df_filtered["priority"].value_counts())

    # Time-series: tickets created per day
    if "created_date" in df_filtered.columns:
        st.markdown("**Tickets created (by day)**")
        ts = df_filtered.groupby(pd.to_datetime(df_filtered["created_date"]).dt.date).size().rename("count").reset_index()
        ts.columns = ["date", "count"]
        if HAS_ALTAIR and not ts.empty:
            ts_chart = alt.Chart(ts).mark_line(point=True).encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("count:Q", title="Tickets"),
                tooltip=["date", "count"]
            ).properties(height=240)
            st.altair_chart(ts_chart, use_container_width=True)
        else:
            st.line_chart(ts.set_index("date")["count"])

    # Histogram of resolution times (hours)
    if "created_date" in df_filtered.columns and "resolved_date" in df_filtered.columns:
        st.markdown("**Resolution time (hours)**")
        mask = df_filtered["resolved_date"].notna() & df_filtered["created_date"].notna()
        if mask.any():
            diffs = (pd.to_datetime(df_filtered.loc[mask, "resolved_date"]) - pd.to_datetime(df_filtered.loc[mask, "created_date"]))
            hours = diffs.dt.total_seconds() / 3600.0
            hist_df = pd.DataFrame({"hours": hours})
            if HAS_ALTAIR:
                hist = alt.Chart(hist_df).transform_bin(
                    "binned_hours", "hours", bin=alt.Bin(maxbins=25)
                ).mark_bar().encode(
                    x=alt.X("binned_hours:Q", title="Resolution hours"),
                    y=alt.Y("count()", title="Tickets"),
                    tooltip=[alt.Tooltip("count()", title="Tickets")]
                ).properties(height=240)
                st.altair_chart(hist, use_container_width=True)
            else:
                st.bar_chart(pd.cut(hours, bins=10).value_counts().sort_index())

with table_col:
    st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True)

with st.expander("Show raw (unfiltered) dataset"):
    st.write(df_display)

st.divider()

# Logout button
if st.button("Log out"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out")
    st.switch_page("Home.py")