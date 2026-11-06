"""
Drug Court Decision Support System (DCDSS) — NIJ Dataset Integration Service
Provides ultra-fast searching, inspection, preview, and field mapping from nijdataset.xlsx.
Strictly preserves the original nijdataset.xlsx file as a read-only historical archive.
"""

import os
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_DIR = BASE_DIR.parent if BASE_DIR.name == "Drug_Court_Decision_Support_System" else BASE_DIR

if (WORKSPACE_DIR / "nijdataset.xlsx").exists():
    EXCEL_PATH = WORKSPACE_DIR / "nijdataset.xlsx"
elif (BASE_DIR / "data" / "nijdataset.xlsx").exists():
    EXCEL_PATH = BASE_DIR / "data" / "nijdataset.xlsx"
else:
    EXCEL_PATH = Path("nijdataset.xlsx").resolve()

CACHE_DB_PATH = BASE_DIR / "data" / "nij_reference.db"
CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Column Categories & Mappings
NIJ_COLUMN_METADATA = [
    {"column": "ID", "category": "Identifier", "type": "Integer", "description": "Historical NIJ challenge subject identifier"},
    {"column": "Gender", "category": "Demographic", "type": "Text (M/F)", "mapsTo": "gender", "description": "Gender of individual"},
    {"column": "Race", "category": "Demographic", "type": "Text (BLACK/WHITE)", "mapsTo": "race", "description": "Racial background"},
    {"column": "Age_at_Release", "category": "Demographic", "type": "Text (e.g. 28-32)", "mapsTo": "age", "description": "Age bracket at release"},
    {"column": "Education_Level", "category": "Demographic", "type": "Text", "mapsTo": "education_level", "description": "Educational attainment"},
    {"column": "Dependents", "category": "Demographic", "type": "Text", "mapsTo": "dependents", "description": "Number of dependents"},
    {"column": "Prison_Offense", "category": "Index Offense", "type": "Text (Drug/Property/Violent)", "mapsTo": "offense_type", "description": "Primary index conviction offense"},
    {"column": "Supervision_Level_First", "category": "Supervision", "type": "Text (Standard/High/Specialized)", "mapsTo": "supervision_level", "description": "Initial supervision tier assigned"},
    {"column": "Supervision_Risk_Score_First", "category": "Supervision", "type": "Integer (1-10)", "mapsTo": "compas_score", "description": "Static risk score assessment"},
    {"column": "Prior_Arrest_Episodes_Felony", "category": "Criminal History", "type": "Integer", "mapsTo": "prior_felony_arrests", "description": "Prior felony arrest count"},
    {"column": "Prior_Arrest_Episodes_Misd", "category": "Criminal History", "type": "Text/Integer", "mapsTo": "prior_misd_arrests", "description": "Prior misdemeanor arrest count"},
    {"column": "Prior_Arrest_Episodes_Violent", "category": "Criminal History", "type": "Text/Integer", "mapsTo": "prior_violent_arrests", "description": "Prior violent arrest count"},
    {"column": "Prior_Arrest_Episodes_Property", "category": "Criminal History", "type": "Integer", "mapsTo": "prior_property_arrests", "description": "Prior property arrest count"},
    {"column": "Prior_Arrest_Episodes_Drug", "category": "Criminal History", "type": "Integer", "mapsTo": "prior_drug_arrests", "description": "Prior drug arrest count"},
    {"column": "Prior_Revocations_Parole", "category": "Criminal History", "type": "Text/Integer", "mapsTo": "prior_revocations", "description": "Prior revocations on parole"},
    {"column": "Prior_Revocations_Probation", "category": "Criminal History", "type": "Text/Integer", "mapsTo": "prior_revocations", "description": "Prior revocations on probation"},
    {"column": "Program_Attendances", "category": "Dynamic Rehabilitation", "type": "Integer", "mapsTo": "attendance_pct", "description": "Verified program attendance sessions"},
    {"column": "Program_UnexcusedAbsences", "category": "Dynamic Rehabilitation", "type": "Integer", "mapsTo": "unexcused_absences", "description": "Unexcused program absence count"},
    {"column": "Percent_Days_Employed", "category": "Dynamic Rehabilitation", "type": "Float (0.0-1.0)", "mapsTo": "employment_pct", "description": "Employment percentage during supervision"},
    {"column": "Jobs_Per_Year", "category": "Dynamic Rehabilitation", "type": "Float", "mapsTo": "jobs_per_year", "description": "Job turnover and stability rate"},
    {"column": "DrugTests_THC_Positive", "category": "Compliance", "type": "Float/Integer", "mapsTo": "drug_screening_compliance", "description": "Positive THC screen count"},
    {"column": "DrugTests_Cocaine_Positive", "category": "Compliance", "type": "Float/Integer", "mapsTo": "drug_screening_compliance", "description": "Positive Cocaine screen count"},
    {"column": "Recidivism_Within_3years", "category": "Target Outcome", "type": "Text (Yes/No)", "description": "3-Year recidivism outcome indicator"}
]

def ensure_cache_db():
    """Builds a local fast SQLite cache of nijdataset.xlsx if not present."""
    if not CACHE_DB_PATH.exists():
        print(f"[*] Building NIJ fast reference index from {EXCEL_PATH} ...")
        df = pd.read_excel(EXCEL_PATH)
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        conn = sqlite3.connect(str(CACHE_DB_PATH))
        df.to_sql("nij_cases", conn, if_exists="replace", index=False)
        # Create indexes on search fields
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nij_id ON nij_cases(ID);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nij_gender ON nij_cases(Gender);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nij_race ON nij_cases(Race);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nij_offense ON nij_cases(Prison_Offense);")
        conn.commit()
        conn.close()
        print(f"[✓] Fast NIJ reference index created at: {CACHE_DB_PATH}")

def get_nij_columns():
    """Returns metadata for all available NIJ columns and mapping documentation."""
    return {
        "dataset": "NIJ Recidivism Challenge 2021 Benchmark",
        "originalFile": "nijdataset.xlsx",
        "totalColumns": len(NIJ_COLUMN_METADATA),
        "columns": NIJ_COLUMN_METADATA,
        "mappingDocumentation": "Features mapped directly to Drug Court participant intake schema. Protected attributes and unmapped fields are documented for fairness governance."
    }

def search_nij_records(
    query: Optional[str] = None,
    gender: Optional[str] = None,
    race: Optional[str] = None,
    offense: Optional[str] = None,
    page: int = 1,
    limit: int = 15
):
    """Fast paginated search through the 25,835 historical NIJ benchmark cases."""
    ensure_cache_db()
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    conditions = []
    params = []

    if query:
        q = f"%{query.strip()}%"
        conditions.append("(CAST(ID AS TEXT) LIKE ? OR Age_at_Release LIKE ? OR Prison_Offense LIKE ?)")
        params.extend([q, q, q])

    if gender and gender != "All":
        conditions.append("Gender = ?")
        params.append("M" if gender.upper().startswith("M") else "F")

    if race and race != "All":
        conditions.append("Race LIKE ?")
        params.append(f"%{race}%")

    if offense and offense != "All":
        conditions.append("Prison_Offense LIKE ?")
        params.append(f"%{offense}%")

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    # Total count query
    count_sql = f"SELECT COUNT(*) FROM nij_cases{where_clause}"
    cursor.execute(count_sql, params)
    total_records = cursor.fetchone()[0]

    # Paginated select query
    offset = (page - 1) * limit
    select_sql = f"""
    SELECT ID, Gender, Race, Age_at_Release, Education_Level, Prison_Offense,
           Supervision_Risk_Score_First, Supervision_Level_First,
           Prior_Arrest_Episodes_Felony, Prior_Arrest_Episodes_Drug,
           Program_Attendances, Program_UnexcusedAbsences, Percent_Days_Employed,
           Recidivism_Within_3years
    FROM nij_cases
    {where_clause}
    LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    cursor.execute(select_sql, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id": r["ID"],
            "gender": "Male" if r["Gender"] == "M" else "Female",
            "race": r["Race"],
            "age": r["Age_at_Release"],
            "education": r["Education_Level"],
            "offense": r["Prison_Offense"],
            "compasScore": r["Supervision_Risk_Score_First"],
            "supervisionLevel": r["Supervision_Level_First"],
            "priorFelonies": r["Prior_Arrest_Episodes_Felony"],
            "priorDrug": r["Prior_Arrest_Episodes_Drug"],
            "attendance": r["Program_Attendances"],
            "absences": r["Program_UnexcusedAbsences"],
            "employment": round(float(r["Percent_Days_Employed"] or 0) * 100, 1),
            "recidivism": r["Recidivism_Within_3years"]
        })

    return {
        "page": page,
        "limit": limit,
        "totalRecords": total_records,
        "totalPages": (total_records + limit - 1) // limit if limit else 1,
        "records": results
    }

def preview_nij_record(record_id: int):
    """Fetches full historical record and maps compatible fields to participant registration schema."""
    ensure_cache_db()
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM nij_cases WHERE ID = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    raw = dict(row)

    # Parse age bracket to numerical estimate
    age_str = str(raw.get("Age_at_Release", "30")).strip()
    if "-" in age_str:
        parts = age_str.split("-")
        try:
            parsed_age = int((int(parts[0]) + int(parts[1])) / 2)
        except Exception:
            parsed_age = 30
    elif "older" in age_str.lower() or "48" in age_str:
        parsed_age = 50
    else:
        parsed_age = 28

    # Parse dependents
    dep_str = str(raw.get("Dependents", "0")).strip()
    parsed_dep = 3 if "3" in dep_str else (1 if "1" in dep_str else (2 if "2" in dep_str else 0))

    # Parse priors
    def parse_int(val, default=0):
        try:
            return int(str(val).split()[0].replace("+", "").replace("or", "").strip())
        except Exception:
            return default

    # Mapped values for Add Participant form
    mapped_form_data = {
        "fullName": f"Imported Candidate #{raw['ID']}",
        "age": parsed_age,
        "gender": "Male" if raw.get("Gender") == "M" else "Female",
        "race": str(raw.get("Race", "Other")).capitalize(),
        "educationLevel": str(raw.get("Education_Level", "High School Diploma")),
        "dependents": parsed_dep,
        "offenseType": f"{raw.get('Prison_Offense', 'General Offense')} Related Offense",
        "programPhase": "Phase I",
        "supervisionLevel": f"Level {min(4, max(1, parse_int(raw.get('Supervision_Risk_Score_First', 2))))}",
        "priorFelonyArrests": parse_int(raw.get("Prior_Arrest_Episodes_Felony", 0)),
        "priorMisdArrests": parse_int(raw.get("Prior_Arrest_Episodes_Misd", 0)),
        "priorViolentArrests": parse_int(raw.get("Prior_Arrest_Episodes_Violent", 0)),
        "priorPropertyArrests": parse_int(raw.get("Prior_Arrest_Episodes_Property", 0)),
        "priorDrugArrests": parse_int(raw.get("Prior_Arrest_Episodes_Drug", 0)),
        "priorRevocations": parse_int(raw.get("Prior_Revocations_Parole", 0)) + parse_int(raw.get("Prior_Revocations_Probation", 0)),
        "attendancePct": min(100.0, float(parse_int(raw.get("Program_Attendances", 10)) * 10)),
        "unexcusedAbsences": parse_int(raw.get("Program_UnexcusedAbsences", 1)),
        "employmentPct": round(float(raw.get("Percent_Days_Employed", 0.0) or 0.0) * 100, 1),
        "jobsPerYear": round(float(raw.get("Jobs_Per_Year", 0.5) or 0.5), 1),
        "treatmentStatus": "In Treatment",
        "compasScore": parse_int(raw.get("Supervision_Risk_Score_First", 5)),
        "notes": f"Historical profile referenced from NIJ Dataset Record #{raw['ID']}. Verified index offense: {raw.get('Prison_Offense')}."
    }

    return {
        "recordId": raw["ID"],
        "rawFields": raw,
        "mappedFields": mapped_form_data,
        "source": "NIJ Recidivism Challenge Historical Reference Dataset",
        "provenanceMetadata": {
            "dataset": "nijdataset.xlsx",
            "sheet": "nij-challenge2021_full_dataset",
            "rowId": raw["ID"],
            "status": "Reference Only — Not Registered in Active Docket Until Saved"
        }
    }

if __name__ == "__main__":
    ensure_cache_db()
    res = search_nij_records(limit=2)
    print("Search test:", res["totalRecords"], "total records found.")
    prev = preview_nij_record(1)
    print("Preview test:", prev["mappedFields"]["fullName"], "| Age:", prev["mappedFields"]["age"])
