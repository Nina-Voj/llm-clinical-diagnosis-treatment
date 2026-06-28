import os
import duckdb
import time

MIMIC_ROOT = r"YOUR_PATH_TO_MIMIC_DATA"
DB_PATH = r"YOUR_PATH_TO_OUTPUT_DATABASE"

hosp_dir = os.path.join(MIMIC_ROOT, "hosp")
icu_dir = os.path.join(MIMIC_ROOT, "icu")

print("Initializing DuckDB database...")
print(f"Database will be saved to: {DB_PATH}")

# If the file exists, we will overwrite/append it. DuckDB handles this.
con = duckdb.connect(DB_PATH)

tables = {
    # Core demographic & stay
    "patients": os.path.join(hosp_dir, "patients.csv.gz"),
    "admissions": os.path.join(hosp_dir, "admissions.csv.gz"),
    "icustays": os.path.join(icu_dir, "icustays.csv.gz"),
    
    # Diagnoses
    "diagnoses_icd": os.path.join(hosp_dir, "diagnoses_icd.csv.gz"),
    "d_icd_diagnoses": os.path.join(hosp_dir, "d_icd_diagnoses.csv.gz"),
    
    # Labs & Meds
    "labevents": os.path.join(hosp_dir, "labevents.csv.gz"),
    "d_labitems": os.path.join(hosp_dir, "d_labitems.csv.gz"),
    "prescriptions": os.path.join(hosp_dir, "prescriptions.csv.gz"),
    
    # New additions
    "omr": os.path.join(hosp_dir, "omr.csv.gz"),
    "procedures_icd": os.path.join(hosp_dir, "procedures_icd.csv.gz"),
    "d_icd_procedures": os.path.join(hosp_dir, "d_icd_procedures.csv.gz"),
    "microbiologyevents": os.path.join(hosp_dir, "microbiologyevents.csv.gz"),
    "d_items": os.path.join(icu_dir, "d_items.csv.gz"),
}

for table_name, csv_path in tables.items():
    print(f"\nImporting {table_name} from {os.path.basename(csv_path)}...")
    start = time.time()
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path.replace(os.sep, '/')}')")
    count = con.query(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"Done in {time.time() - start:.2f} seconds. Inserted {count:,} rows.")

print(f"\nImporting FILTERED chartevents from chartevents.csv.gz (Vitals only)...")
start = time.time()
con.execute("DROP TABLE IF EXISTS chartevents")
vital_ids = "(220045, 220277, 223761, 223762, 220210, 220179, 220180, 220181)"
chart_path = os.path.join(icu_dir, "chartevents.csv.gz").replace(os.sep, '/')
con.execute(f"CREATE TABLE chartevents AS SELECT * FROM read_csv_auto('{chart_path}') WHERE itemid IN {vital_ids}")
count = con.query("SELECT COUNT(*) FROM chartevents").fetchone()[0]
print(f"Done in {time.time() - start:.2f} seconds. Inserted {count:,} filtered rows.")

print("\nCreating indexes to speed up queries...")
start = time.time()
con.execute("CREATE INDEX IF NOT EXISTS idx_patients_sub ON patients (subject_id)")
con.execute("CREATE INDEX IF NOT EXISTS idx_admissions_sub_hadm ON admissions (subject_id, hadm_id)")
con.execute("CREATE INDEX IF NOT EXISTS idx_diagnoses_hadm ON diagnoses_icd (hadm_id)")
con.execute("CREATE INDEX IF NOT EXISTS idx_labevents_hadm ON labevents (hadm_id)")
con.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_hadm ON prescriptions (hadm_id)")
con.execute("CREATE INDEX IF NOT EXISTS idx_icustays_hadm ON icustays (hadm_id)")

# New indexes
con.execute("CREATE INDEX IF NOT EXISTS idx_omr_sub ON omr (subject_id)")
con.execute("CREATE INDEX IF NOT EXISTS idx_procedures_hadm ON procedures_icd (hadm_id)")
con.execute("CREATE INDEX IF NOT EXISTS idx_micro_hadm ON microbiologyevents (hadm_id)")
con.execute("CREATE INDEX IF NOT EXISTS idx_chartevents_hadm ON chartevents (hadm_id)")

print(f"Indexes created in {time.time() - start:.2f} seconds.")

con.close()
print("\nDatabase initialization complete! You can now run the pipeline.")
