"""Dataset ingestion, profiling, statistics, and storage service."""
from __future__ import annotations

import io
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from fastapi import HTTPException
from app.config import settings
from app.db.models import UserDataset

logger = logging.getLogger("machinasense.datasets")

def sanitize_filename(filename: str) -> str:
    base = os.path.basename(filename.replace("\\", "/")).split("/")[-1].strip()
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", base).strip(". _")
    return cleaned or "dataset.csv"

def _safe_float(val: Any) -> Optional[float]:
    if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return None

def process_csv_dataset(raw_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Parses and calculates comprehensive statistics and preview from CSV bytes.
    Validates tabular format, calculates column metadata and distribution metrics.
    """
    if not raw_bytes or len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Dataset file is empty (0 bytes).")

    if len(raw_bytes) > settings.MAX_FILE_SIZE_BYTES:
        max_mb = settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"File exceeds maximum allowed upload limit of {max_mb}MB.")

    # Attempt decoding and reading via pandas
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding="latin1")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="CSV encoding error. Please upload a standard UTF-8 encoded CSV.") from exc
    except pd.errors.EmptyDataError as exc:
        raise HTTPException(status_code=400, detail="CSV file contains no data or header rows.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Malformed CSV file. Please verify RFC 4180 tabular structure.") from exc

    if df.empty or len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="Dataset contains no readable records or columns.")

    row_count = int(len(df))
    col_count = int(len(df.columns))

    # Calculate overall dataset statistics
    duplicate_count = int(df.duplicated().sum())
    missing_count = int(df.isna().sum().sum())
    try:
        memory_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        memory_bytes = int(len(raw_bytes))

    columns_meta: List[Dict[str, Any]] = []
    numeric_col_count = 0
    categorical_col_count = 0

    for col in df.columns:
        col_name = str(col).strip()
        series = df[col]
        is_num = bool(pd.api.types.is_numeric_dtype(series))
        col_missing = int(series.isna().sum())
        col_unique = int(series.nunique(dropna=True))

        numeric_stats = None
        categorical_stats = None

        if is_num:
            numeric_col_count += 1
            valid_vals = series.dropna()
            if len(valid_vals) > 0:
                numeric_stats = {
                    "mean": _safe_float(valid_vals.mean()) or 0.0,
                    "std": _safe_float(valid_vals.std(ddof=0)) if len(valid_vals) > 1 else 0.0,
                    "min": _safe_float(valid_vals.min()) or 0.0,
                    "max": _safe_float(valid_vals.max()) or 0.0,
                    "median": _safe_float(valid_vals.median()) or 0.0,
                    "q25": _safe_float(valid_vals.quantile(0.25)) or 0.0,
                    "q75": _safe_float(valid_vals.quantile(0.75)) or 0.0,
                }
        else:
            categorical_col_count += 1
            vc = series.value_counts(dropna=True)
            top_val = str(vc.index[0]) if len(vc) > 0 else ""
            top_freq = int(vc.iloc[0]) if len(vc) > 0 else 0
            categorical_stats = {
                "uniqueCount": col_unique,
                "topValue": top_val,
                "topFrequency": top_freq,
            }

        columns_meta.append({
            "name": col_name,
            "dtype": str(series.dtype),
            "missingCount": col_missing,
            "uniqueCount": col_unique,
            "isNumeric": is_num,
            "numericStats": numeric_stats,
            "categoricalStats": categorical_stats,
        })

    # Generate bounded preview rows (up to 50 rows)
    preview_limit = min(50, row_count)
    preview_df = df.iloc[:preview_limit].copy()

    preview_rows: List[Dict[str, Any]] = []
    for _, row in preview_df.iterrows():
        row_dict: Dict[str, Any] = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val) or val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                row_dict[str(col).strip()] = None
            elif isinstance(val, (np.integer, int)):
                row_dict[str(col).strip()] = int(val)
            elif isinstance(val, (np.floating, float)):
                row_dict[str(col).strip()] = float(val)
            elif isinstance(val, (np.bool_, bool)):
                row_dict[str(col).strip()] = bool(val)
            else:
                row_dict[str(col).strip()] = str(val)
        preview_rows.append(row_dict)

    preview_meta = {
        "columns": [str(c).strip() for c in df.columns],
        "rows": preview_rows,
        "totalRows": row_count,
        "previewRows": len(preview_rows),
    }

    stats_meta = {
        "rowCount": row_count,
        "columnCount": col_count,
        "numericColumnCount": numeric_col_count,
        "categoricalColumnCount": categorical_col_count,
        "missingValueCount": missing_count,
        "duplicateRowCount": duplicate_count,
        "memorySizeBytes": memory_bytes,
    }

    return {
        "statistics": stats_meta,
        "columns": columns_meta,
        "preview": preview_meta,
        "rowCount": row_count,
        "columnCount": col_count,
        "numericColumnCount": numeric_col_count,
        "categoricalColumnCount": categorical_col_count,
        "missingValueCount": missing_count,
        "duplicateRowCount": duplicate_count,
        "memorySizeBytes": memory_bytes,
    }

def save_dataset_file(user_id: str, dataset_id: str, content: bytes) -> str:
    """Saves raw dataset CSV to persistent user storage directory."""
    user_dir = os.path.join(settings.DATASETS_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, f"{dataset_id}.csv")
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path

def delete_dataset_file(file_path: str) -> None:
    """Removes stored dataset file safely if present."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as exc:
        logger.warning("Could not delete dataset file at %s: %s", file_path, exc)

def serialize_dataset_view(d: UserDataset) -> Dict[str, Any]:
    """Serializes UserDataset into the frontend Dataset contract."""
    columns = json.loads(d.columns_json) if d.columns_json else []
    preview = json.loads(d.preview_json) if d.preview_json else None
    stats = {
        "rowCount": d.row_count or 0,
        "columnCount": d.column_count or 0,
        "numericColumnCount": d.numeric_column_count or 0,
        "categoricalColumnCount": d.categorical_column_count or 0,
        "missingValueCount": d.missing_value_count or 0,
        "duplicateRowCount": d.duplicate_row_count or 0,
        "memorySizeBytes": d.memory_size_bytes or 0,
    }
    return {
        "id": d.id,
        "name": d.name,
        "fileName": d.original_filename,
        "fileSize": d.file_size,
        "status": d.status,
        "statistics": stats,
        "columns": columns,
        "preview": preview,
        "uploadedAt": (d.created_at).isoformat() if d.created_at else "",
    }

def serialize_dataset_list_item(d: UserDataset) -> Dict[str, Any]:
    """Serializes UserDataset into lightweight frontend DatasetListItem contract."""
    return {
        "id": d.id,
        "name": d.name,
        "fileName": d.original_filename,
        "fileSize": d.file_size,
        "status": d.status,
        "rowCount": d.row_count,
        "columnCount": d.column_count,
        "uploadedAt": (d.created_at).isoformat() if d.created_at else "",
    }
