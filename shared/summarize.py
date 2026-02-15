"""
Strategy D summarizer for entity resolution.

Generates structured, diff-friendly patient summaries grouped by year,
designed for pairwise comparison by the fine-tuned classifier.
"""

import pandas as pd

from shared.medical_records import get_patient_records

INSTRUCTION = (
    "You are a medical record matching expert. Compare these two patient "
    "medical records and determine if they belong to the same patient based "
    "only on their clinical history.\n\n"
    "Record A:\n{summary_a}\n\n"
    "Record B:\n{summary_b}\n\n"
    "Are these the same patient? Answer only 'True' or 'False'."
)


def summarize_diff_friendly(patient_id, facility_id, medical_records):
    """Structured for pairwise comparison, grouped by year. Target ~800 tokens."""
    records = get_patient_records(patient_id, facility_id, medical_records)
    return summarize_diff_friendly_from_records(records)


def summarize_diff_friendly_from_records(records):
    """Structured for pairwise comparison, grouped by year. Target ~800 tokens."""
    sections = []

    # CONDITIONS - grouped by onset year
    cond_df = records.get("conditions")
    if cond_df is not None:
        cond_df = cond_df.copy()
        cond_df["year"] = pd.to_datetime(cond_df["START"], errors="coerce").dt.year
        cond_df["is_ongoing"] = cond_df["STOP"].isna() | (
            cond_df["STOP"].astype(str).str.strip() == ""
        )
        lines = ["CONDITIONS:"]
        for year, grp in sorted(cond_df.groupby("year")):
            if pd.isna(year):
                continue
            descs = []
            for _, row in grp.iterrows():
                status = " *" if row["is_ongoing"] else ""
                descs.append(f"{row['DESCRIPTION']}{status}")
            lines.append(f"  {int(year)}: {'; '.join(descs)}")
        sections.append("\n".join(lines))

    # MEDICATIONS - drug (start_year-end_year or ongoing)
    meds_df = records.get("medications")
    if meds_df is not None:
        meds_df = meds_df.copy()
        lines = ["MEDICATIONS:"]
        for desc, grp in meds_df.groupby("DESCRIPTION", sort=False):
            start_dt = pd.to_datetime(grp["START"], errors="coerce").min()
            is_current = (
                grp["STOP"].isna().any()
                | (grp["STOP"].astype(str).str.strip() == "").any()
            )
            if is_current:
                period = (
                    f"{start_dt.year}\u2013ongoing" if pd.notna(start_dt) else "ongoing"
                )
            else:
                end_dt = pd.to_datetime(grp["STOP"], errors="coerce").max()
                if pd.notna(start_dt) and pd.notna(end_dt):
                    period = f"{start_dt.year}\u2013{end_dt.year}"
                else:
                    period = "unknown"
            lines.append(f"- {desc} ({period})")
        sections.append("\n".join(lines))

    # ALLERGIES - flat list
    allg_df = records.get("allergies")
    if allg_df is not None:
        names = sorted(allg_df["DESCRIPTION"].unique())
        sections.append("ALLERGIES: " + "; ".join(names))

    # KEY OBSERVATIONS - latest 2 values per metric
    obs_df = records.get("observations")
    if obs_df is not None:
        obs_df = obs_df.copy()
        obs_df["date_dt"] = pd.to_datetime(obs_df["DATE"], errors="coerce")
        key_obs = [
            "Body Height",
            "Body Weight",
            "Body Mass Index",
            "Systolic Blood Pressure",
            "Diastolic Blood Pressure",
            "Hemoglobin A1c/Hemoglobin.total in Blood",
            "Glucose",
            "Total Cholesterol",
        ]
        lines = ["OBSERVATIONS:"]
        for obs_name in key_obs:
            match = obs_df[
                obs_df["DESCRIPTION"].str.contains(obs_name, case=False, na=False)
            ]
            if match.empty:
                continue
            recent = match.sort_values("date_dt").tail(2)
            vals = []
            for _, row in recent.iterrows():
                v = row.get("VALUE", "")
                u = row.get("UNITS", "")
                d = str(row.get("DATE", ""))[:10]
                if pd.notna(v) and str(v).strip():
                    u_str = f" {u}" if pd.notna(u) and u else ""
                    vals.append(f"{v}{u_str} ({d})")
            if vals:
                lines.append(f"- {obs_name}: {', '.join(vals)}")
        sections.append("\n".join(lines))

    # PROCEDURES - with years, chronological
    proc_df = records.get("procedures")
    if proc_df is not None:
        proc_df = proc_df.copy()
        proc_df["year"] = pd.to_datetime(proc_df["START"], errors="coerce").dt.year
        lines = ["PROCEDURES:"]
        for desc, grp in proc_df.groupby("DESCRIPTION", sort=False):
            years = sorted(grp["year"].dropna().unique())
            year_strs = [str(int(y)) for y in years]
            lines.append(f"- {desc} ({', '.join(year_strs)})")
        sections.append("\n".join(lines))

    return "\n\n".join(sections) if sections else "No clinical records available."
