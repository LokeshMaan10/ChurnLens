"""Tenure and demographic features from the snapshot's members file.

Users who registered on/after the cutoff get every member attribute NULL and
member_info_missing = 1, exactly like users with no members row; otherwise a NULL
tenure next to a present city would reveal a post-cutoff registration.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path


def members_sql(snap_dir: Path, cutoff: dt.date) -> str:
    """SQL returning msno + tenure and demographic features."""
    s, c = snap_dir.as_posix(), f"DATE '{cutoff.isoformat()}'"
    usable = "(m.msno IS NOT NULL AND NOT m.registered_after_cutoff " \
             f"AND m.registration_date < {c})"
    return f"""
        WITH lab AS (SELECT msno FROM read_parquet('{s}/labels.parquet')),
        m AS (SELECT * FROM read_parquet('{s}/members.parquet'))
        SELECT l.msno,
               CASE WHEN {usable} THEN date_diff('day', m.registration_date, {c}) END
                   AS tenure_days,
               CASE WHEN {usable} THEN CAST(m.registered_via AS VARCHAR) END AS registered_via,
               (NOT {usable})::INT AS member_info_missing,
               CASE WHEN {usable} THEN m.age::DOUBLE END AS age,
               CASE WHEN {usable} THEN COALESCE(m.gender, 'unknown') END AS gender,
               CASE WHEN {usable} THEN CAST(m.city AS VARCHAR) END AS city
        FROM lab l LEFT JOIN m USING (msno)"""


def rfm_sql(rfm_file: Path) -> str:
    """SQL returning msno + the Phase 4 RFM outputs used as features."""
    return f"""
        SELECT msno, r_score, f_score, m_score, personal_recency::DOUBLE AS personal_recency,
               segment AS rfm_segment, cluster AS rfm_group
        FROM read_parquet('{rfm_file.as_posix()}')"""
