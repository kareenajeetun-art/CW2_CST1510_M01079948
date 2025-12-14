"""
Datasets Metadata Dashboard (DB + CSV)
 - Loads datasets_metadata from the SQLite DB (DATA/intelligence_platform.db)
 - via sqlite3 + pandas.
 - Loads DATA/datasets_metadata.csv directly.
 - Lets the user choose: view DB table / CSV / combined dataset.
 - Displays KPIs, charts, and the table. Allows adding a new dataset metadata entry
   (via DB insert function if available).
"""

import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
import inspect

# page config
st.set_page_config(page_title="Datasets Metadata", layout="wide", page_icon="📚")

# Authentication guard
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.error("You must be logged in to view datasets metadata.")
    if st.button("Go to login"):
        st.switch_page("Home.py")
    st.stop()

# Paths
DB_PATH = Path("DATA") / "intelligence_platform.db"
CSV_PATH = Path("DATA") / "datasets_metadata.csv"
TABLE_NAME = "datasets_metadata"


# Helper: connect to DB
def connect_via_helper(db_path: Path):
    try:
        from app.data.db import connect_database  # type: ignore
        # try to call with path if the helper accepts path, else call without args
        try:
            return connect_database(str(db_path))
        except TypeError:
            return connect_database()
    except Exception:
        raise


def connect_fallback(db_path: Path):
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return conn


@st.cache_resource
def get_connection(db_path: Path = DB_PATH) -> Optional[sqlite3.Connection]:
    """
    Return a cached DB connection resource or None if DB not present/openable.
    """
    if not db_path.exists():
        st.warning(f"Database file not found at {db_path}. DB features will be disabled.")
        return None
    try:
        conn = connect_via_helper(db_path)
        return conn
    except Exception:
        try:
            conn = connect_fallback(db_path)
            return conn
        except Exception as e:
            st.error(f"Unable to open DB: {e}")
            return None


# Load DB table into DataFrame
@st.cache_data(ttl=60)
def load_db_table(table_name: str = TABLE_NAME) -> pd.DataFrame:
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        # try to show available tables to help debugging
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


# Try to import insert function to allow adding into DB 
def try_get_insert_function() -> Optional[Callable]:
    """
    If you've implemented app.data.datasets.insert_dataset_metadata, return it.
    """
    try:
        from app.data.datasets import insert_dataset_metadata  # type: ignore
        return insert_dataset_metadata
    except Exception:
        return None


# Normalization helper (for safe combining)
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # trim columns
    df.columns = [c.strip() for c in df.columns]
    # map likely column name variations to canonical names
    names = [c.lower() for c in df.columns]
    colmap = {}
    # dataset name
    for alt in ("name", "dataset", "dataset_name", "title"):
        if alt in names and "dataset_name" not in names:
            orig = df.columns[names.index(alt)]
            colmap[orig] = "dataset_name"
            break
    # category
    for alt in ("category", "type", "group"):
        if alt in names and "category" not in names:
            orig = df.columns[names.index(alt)]
            colmap[orig] = "category"
            break
    # source
    for alt in ("source", "uploaded_by", "owner"):
        if alt in names and "source" not in names:
            orig = df.columns[names.index(alt)]
            colmap[orig] = "source"
            break
    # last_updated / upload_date
    for alt in ("last_updated", "upload_date", "date", "updated_at"):
        if alt in names and "last_updated" not in names:
            orig = df.columns[names.index(alt)]
            colmap[orig] = "last_updated"
            break
    # record_count / rows
    for alt in ("record_count", "rows", "count"):
        if alt in names and "record_count" not in names:
            orig = df.columns[names.index(alt)]
            colmap[orig] = "record_count"
            break
    # file size
    for alt in ("file_size_mb", "size_mb", "filesize"):
        if alt in names and "file_size_mb" not in names:
            orig = df.columns[names.index(alt)]
            colmap[orig] = "file_size_mb"
            break

    if colmap:
        df = df.rename(columns=colmap)

    # coerce types
    if "last_updated" in df.columns:
        try:
            df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
        except Exception:
            pass

    if "record_count" in df.columns:
        try:
            df["record_count"] = pd.to_numeric(df["record_count"], errors="coerce").fillna(0).astype(int)
        except Exception:
            pass

    return df


# Load data
db_df = load_db_table()
csv_df = load_csv(CSV_PATH)

db_df_norm = normalize_df(db_df)
csv_df_norm = normalize_df(csv_df)

# UI: choose source
st.title("📚 Datasets Metadata")
st.subheader(f"Hello, {st.session_state.username} — choose source to view")

source = st.radio(
    "Data source",
    options=["Database table (DB)", "CSV file", "Combined (DB + CSV)"],
    index=0,
)

if source == "Database table (DB)":
    df_display = db_df_norm.copy()
elif source == "CSV file":
    df_display = csv_df_norm.copy()
else:
    combined = pd.concat([db_df_norm, csv_df_norm], ignore_index=True, sort=False)
    # drop duplicates by id or dataset_name+last_updated if available
    if "id" in combined.columns:
        combined = combined.drop_duplicates(subset=["id"])
    else:
        keys = []
        if "dataset_name" in combined.columns:
            keys.append("dataset_name")
        if "last_updated" in combined.columns:
            keys.append("last_updated")
        if keys:
            combined = combined.drop_duplicates(subset=keys)
        else:
            combined = combined.drop_duplicates()
    df_display = combined.reset_index(drop=True)

# KPIs
total = len(df_display)
large_count = int((df_display.get("record_count", 0) >= 100000).sum()) if "record_count" in df_display.columns else 0
unique_categories = int(df_display["category"].nunique()) if "category" in df_display.columns else 0

col1, col2, col3 = st.columns(3)
col1.metric("Displayed datasets", total)
col2.metric("Large datasets (≥100k rows)", large_count)
col3.metric("Categories", unique_categories)

st.divider()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    # date range for last_updated
    if "last_updated" in df_display.columns and not df_display["last_updated"].isna().all():
        min_date = pd.to_datetime(df_display["last_updated"]).min().date()
        max_date = pd.to_datetime(df_display["last_updated"]).max().date()
        date_range = st.date_input("Last updated range", value=(min_date, max_date))
    else:
        date_range = None

    # category
    if "category" in df_display.columns:
        avail = sorted(df_display["category"].dropna().unique().tolist())
        category_sel = st.multiselect("Category", options=avail, default=avail)
    else:
        category_sel = []

    # source filter
    if "source" in df_display.columns:
        avail = sorted(df_display["source"].dropna().unique().tolist())
        source_sel = st.multiselect("Source", options=avail, default=avail)
    else:
        source_sel = []

    # record_count slider (if present)
    if "record_count" in df_display.columns:
        max_rows = int(df_display["record_count"].max() or 0)
        min_rows = int(df_display["record_count"].min() or 0)
        rows_range = st.slider("Record count range", min_value=min_rows, max_value=max_rows, value=(min_rows, max_rows))
    else:
        rows_range = None

    if st.button("Refresh data"):
        load_db_table.clear()
        load_csv.clear()
        st.rerun()

# Apply filters
df_filtered = df_display.copy()

if date_range is not None and "last_updated" in df_filtered.columns:
    start_date, end_date = date_range
    df_filtered = df_filtered[
        (pd.to_datetime(df_filtered["last_updated"]).dt.date >= start_date)
        & (pd.to_datetime(df_filtered["last_updated"]).dt.date <= end_date)
    ]

if category_sel and "category" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["category"].isin(category_sel)]

if source_sel and "source" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["source"].isin(source_sel)]

if rows_range is not None and "record_count" in df_filtered.columns:
    lo, hi = rows_range
    df_filtered = df_filtered[(df_filtered["record_count"] >= lo) & (df_filtered["record_count"] <= hi)]

# Charts + table
st.subheader("Datasets Overview")

chart_col, table_col = st.columns((1, 2))

with chart_col:
    if "category" in df_filtered.columns and not df_filtered["category"].isna().all():
        cat_counts = df_filtered["category"].value_counts()
        st.bar_chart(cat_counts)

    if "record_count" in df_filtered.columns and not df_filtered["record_count"].isna().all():
        # histogram-like plot via value_counts by binning
        try:
            s = df_filtered["record_count"].dropna()
            bins = pd.cut(s, bins=10)
            counts = bins.value_counts().sort_index()
            st.bar_chart(counts)
        except Exception:
            pass

    if "last_updated" in df_filtered.columns and not df_filtered["last_updated"].isna().all():
        ts = df_filtered.groupby(pd.to_datetime(df_filtered["last_updated"]).dt.date).size().rename("count")
        st.line_chart(ts)

with table_col:
    st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True)

with st.expander("Show raw (unfiltered) dataset"):
    st.write(df_display)

st.divider()

# Insert new dataset metadata (only if DB connection + insert function available)
insert_func = try_get_insert_function()
conn = get_connection()
can_insert = (conn is not None) and (insert_func is not None)

st.subheader("Add new dataset metadata")
if can_insert:
    with st.form("add_dataset_metadata"):
        name = st.text_input("Dataset name")
        category = st.text_input("Category", value="General")
        source = st.text_input("Source / Uploaded by")
        last_updated = st.date_input("Last updated", value=datetime.today().date())
        record_count = st.number_input("Record count (rows)", min_value=0, value=0, step=1)
        file_size_mb = st.number_input("File size (MB)", min_value=0.0, value=0.0, format="%.2f")
        submitted = st.form_submit_button("Add to DB")

        if submitted:
            if not name.strip():
                st.error("Dataset name required.")
            else:
                try:
                    # attempt to call insert function. it may accept a conn or not.
                    sig = inspect.signature(insert_func)
                    params = sig.parameters
                    # common variants:
                    #  - insert_dataset_metadata(dataset_name, category, source, last_updated, record_count=None, file_size_mb=None)
                    #  - insert_dataset_metadata(conn, dataset_name, ...)
                    if len(params) >= 1:
                        # detect if first param is 'conn' (simple heuristic by name/type)
                        first_param = next(iter(params.values()))
                        if first_param.name in ("conn", "connection", "db_conn"):
                            # call with conn then rest
                            insert_func(conn, name.strip(), category.strip(), source.strip(), str(last_updated), int(record_count), float(file_size_mb))
                        else:
                            # call without conn
                            insert_func(name.strip(), category.strip(), source.strip(), str(last_updated), int(record_count), float(file_size_mb))
                    else:
                        # fallback: try calling without conn
                        insert_func(name.strip(), category.strip(), source.strip(), str(last_updated), int(record_count), float(file_size_mb))

                    st.success("Inserted dataset metadata.")
                    # clear caches and reload
                    load_db_table.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Insert failed: {e}")
else:
    st.info(
        "Add-to-DB is disabled. To enable: ensure DATA/intelligence_platform.db exists, "
        "and implement `insert_dataset_metadata` in app/data/datasets.py (or adjust helper import)."
    )

# Logout button
st.divider()
if st.button("Log out"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out")
    st.switch_page("Home.py")