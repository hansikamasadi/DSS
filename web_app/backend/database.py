"""
Drug Court Decision Support System (DCDSS) — Database Engine
SQLite-backed persistent database with full relational schema, seeding, and migrations.
"""

import os
import sqlite3
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "drug_court.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # 1. USERS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        badge_id TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        email TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        pin TEXT NOT NULL,
        role TEXT NOT NULL,
        department TEXT NOT NULL,
        is_active INTEGER DEFAULT 1,
        created_at TEXT NOT NULL
    );
    """)

    # 2. PARTICIPANTS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS participants (
        id TEXT PRIMARY KEY,
        docket_number TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        race TEXT NOT NULL,
        education_level TEXT NOT NULL,
        dependents INTEGER NOT NULL,
        offense_type TEXT NOT NULL,
        program_phase TEXT NOT NULL,
        supervision_level TEXT NOT NULL,
        enrollment_date TEXT NOT NULL,
        assigned_judge TEXT NOT NULL,
        assigned_case_manager TEXT NOT NULL,
        assigned_probation_officer TEXT NOT NULL,
        prior_felony_arrests INTEGER DEFAULT 0,
        prior_misd_arrests INTEGER DEFAULT 0,
        prior_violent_arrests INTEGER DEFAULT 0,
        prior_property_arrests INTEGER DEFAULT 0,
        prior_drug_arrests INTEGER DEFAULT 0,
        prior_revocations INTEGER DEFAULT 0,
        attendance_pct REAL DEFAULT 0.0,
        unexcused_absences INTEGER DEFAULT 0,
        employment_pct REAL DEFAULT 0.0,
        jobs_per_year REAL DEFAULT 0.0,
        drug_screening_compliance REAL DEFAULT 100.0,
        treatment_status TEXT DEFAULT 'In Treatment',
        compas_score INTEGER DEFAULT 5,
        risk_probability REAL DEFAULT 50.0,
        risk_category TEXT DEFAULT 'Medium Risk',
        notes TEXT,
        imported_from_nij INTEGER DEFAULT 0,
        nij_record_id TEXT,
        is_archived INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    # 3. TREATMENT PLANS TABLE (Case Manager)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS treatment_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id TEXT NOT NULL,
        case_manager_id TEXT NOT NULL,
        plan_name TEXT NOT NULL,
        clinical_goals TEXT NOT NULL,
        prescribed_interventions TEXT NOT NULL,
        counseling_frequency TEXT NOT NULL,
        target_attendance REAL NOT NULL,
        target_employment REAL NOT NULL,
        status TEXT DEFAULT 'Active',
        created_at TEXT NOT NULL,
        FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)

    # 4. COUNSELING SESSIONS TABLE (Case Manager)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS counseling_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id TEXT NOT NULL,
        case_manager_id TEXT NOT NULL,
        session_date TEXT NOT NULL,
        session_type TEXT NOT NULL,
        attendance_status TEXT NOT NULL,
        clinical_notes TEXT NOT NULL,
        progress_rating INTEGER DEFAULT 3,
        FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)

    # 5. ATTENDANCE RECORDS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id TEXT NOT NULL,
        date TEXT NOT NULL,
        program_type TEXT NOT NULL,
        status TEXT NOT NULL,
        verified_by_officer TEXT,
        FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)

    # 6. EMPLOYMENT RECORDS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employment_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id TEXT NOT NULL,
        employer_name TEXT NOT NULL,
        job_title TEXT NOT NULL,
        hours_per_week REAL DEFAULT 40.0,
        hourly_wage REAL DEFAULT 18.0,
        verification_status TEXT DEFAULT 'Verified',
        reported_date TEXT NOT NULL,
        FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)

    # 7. DRUG SCREENINGS TABLE (Probation Officer)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS drug_screenings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id TEXT NOT NULL,
        officer_id TEXT NOT NULL,
        screening_date TEXT NOT NULL,
        test_type TEXT NOT NULL,
        result TEXT NOT NULL,
        detected_substances TEXT,
        sanction_flag INTEGER DEFAULT 0,
        FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)

    # 8. PROBATION VIOLATIONS TABLE (Probation Officer)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS probation_violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id TEXT NOT NULL,
        officer_id TEXT NOT NULL,
        violation_date TEXT NOT NULL,
        violation_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        description TEXT NOT NULL,
        reported_to_judge INTEGER DEFAULT 1,
        status TEXT DEFAULT 'Reported',
        FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)

    # 9. FIELD VISITS TABLE (Probation Officer)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS field_visits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id TEXT NOT NULL,
        officer_id TEXT NOT NULL,
        scheduled_date TEXT NOT NULL,
        visit_type TEXT NOT NULL,
        status TEXT NOT NULL,
        notes TEXT,
        FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)

    # 10. JUDICIAL DECISIONS TABLE (Judge)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS judicial_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        log_id TEXT NOT NULL,
        participant_id TEXT NOT NULL,
        judge_id TEXT NOT NULL,
        judge_name TEXT NOT NULL,
        hearing_date TEXT NOT NULL,
        decision_type TEXT NOT NULL,
        ordered_interventions TEXT NOT NULL,
        field_notes TEXT NOT NULL,
        crypto_hash TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY(participant_id) REFERENCES participants(id)
    );
    """)

    # 11. AUDIT LOGS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        log_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        user_id TEXT NOT NULL,
        user_name TEXT NOT NULL,
        user_role TEXT NOT NULL,
        action TEXT NOT NULL,
        participant_id TEXT,
        participant_name TEXT,
        details TEXT NOT NULL,
        crypto_hash TEXT NOT NULL
    );
    """)

    # 12. NOTIFICATIONS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_role TEXT NOT NULL,
        title TEXT NOT NULL,
        message TEXT NOT NULL,
        type TEXT DEFAULT 'info',
        is_read INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    );
    """)

    conn.commit()

    # Seed Initial Users if Empty
    cursor.execute("SELECT COUNT(*) FROM users;")
    if cursor.fetchone()[0] == 0:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        seed_users = [
            ("JUD-8492-EAST", "Hon. Evelyn Carter", "judge.carter@drugcourt.gov", hash_password("judge123"), "8492", "Judge", "Bench #4 Criminal Division", 1, now),
            ("CM-4019-CLIN", "David Chen, MSW", "david.chen@drugcourt.gov", hash_password("case123"), "4019", "Case Manager", "Clinical Rehabilitation Unit", 1, now),
            ("PO-2281-DIST", "Officer Frank Miller", "f.miller@probation.gov", hash_password("probation123"), "2281", "Probation Officer", "Field Supervision District #2", 1, now),
            ("ADM-1001-ROOT", "Sarah Connor", "s.connor@courtadmin.gov", hash_password("admin123"), "1001", "Administrator", "Judicial Information Technology", 1, now)
        ]
        cursor.executemany("""
        INSERT INTO users (badge_id, full_name, email, password_hash, pin, role, department, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, seed_users)
        conn.commit()
        print("[✓] Seeded default role-based users.")

    # Seed Initial Participants if Empty
    cursor.execute("SELECT COUNT(*) FROM participants;")
    if cursor.fetchone()[0] == 0:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        seed_participants = [
            (
                "DC-0812", "2026-CR-09214", "Marcus Vance", 31, "Male", "Black", "High School Grad", 1,
                "Controlled Substance Distribution (Class B)", "Phase II", "Level 3", "2026-03-15",
                "Hon. Evelyn Carter", "David Chen, MSW", "Officer Frank Miller",
                4, 3, 1, 1, 3, 2, 12.0, 5, 0.0, 0.0, 60.0, "Sanction Pending", 7, 68.4, "High Risk",
                "Urinalysis compliance drop & attendance fell to 12% in last 30 days. Candidate for vocational intervention.",
                0, None, 0, now, now
            ),
            (
                "DC-0749", "2026-CR-08731", "Jamal Thomas", 26, "Male", "Black", "Less than High School", 0,
                "Possession with Intent to Distribute", "Phase I", "Level 4", "2026-05-10",
                "Hon. Evelyn Carter", "David Chen, MSW", "Officer Frank Miller",
                5, 4, 1, 2, 4, 2, 18.0, 4, 15.0, 0.3, 50.0, "Sanction Pending", 8, 72.1, "High Risk",
                "Curfew violation & 3 consecutive unexcused program absences.",
                0, None, 0, now, now
            ),
            (
                "DC-0891", "2026-CR-09845", "Chloe Bennett", 28, "Female", "White", "Some College", 2,
                "Prescription Fraud & Possession", "Phase II", "Level 2", "2026-04-01",
                "Hon. Evelyn Carter", "David Chen, MSW", "Officer Frank Miller",
                1, 2, 0, 1, 2, 0, 58.0, 2, 45.0, 1.0, 90.0, "In Treatment", 5, 48.3, "Medium Risk",
                "Missed counseling check-in due to transit disruption; transit card provided.",
                0, None, 0, now, now
            ),
            (
                "DC-0664", "2026-CR-07912", "Derek Washington", 34, "Male", "Black", "High School Grad", 1,
                "Theft II & Substance Abuse", "Phase II", "Level 3", "2026-01-20",
                "Hon. Evelyn Carter", "David Chen, MSW", "Officer Frank Miller",
                3, 4, 0, 3, 2, 1, 35.0, 3, 30.0, 0.7, 75.0, "In Treatment", 6, 64.8, "High Risk",
                "Phase II compliance milestone review.",
                0, None, 0, now, now
            ),
            (
                "DC-0511", "2025-CR-06482", "Elena Rostova", 35, "Female", "White", "Bachelor Degree", 1,
                "Substance Misdemeanor", "Phase III", "Level 1", "2025-08-14",
                "Hon. Evelyn Carter", "David Chen, MSW", "Officer Frank Miller",
                0, 1, 0, 0, 1, 0, 92.0, 0, 85.0, 1.0, 100.0, "Compliant", 2, 18.5, "Low Risk",
                "Fully compliant with outpatient recovery.",
                0, None, 0, now, now
            ),
            (
                "DC-0422", "2025-CR-05391", "Aaliyah Brooks", 27, "Female", "Black", "Associate Degree", 2,
                "Possession of Controlled Substance", "Phase IV", "Level 1", "2025-06-01",
                "Hon. Evelyn Carter", "David Chen, MSW", "Officer Frank Miller",
                0, 2, 0, 0, 2, 0, 96.0, 0, 95.0, 1.0, 100.0, "Compliant", 2, 14.2, "Low Risk",
                "Completed Phase IV curriculum; pending graduation hearing.",
                0, None, 0, now, now
            ),
            (
                "DC-0388", "2025-CR-04877", "Carlos Mendez", 32, "Male", "Hispanic / White", "High School Grad", 0,
                "Drug Paraphernalia & Possession", "Phase II", "Level 2", "2025-09-18",
                "Hon. Evelyn Carter", "David Chen, MSW", "Officer Frank Miller",
                2, 3, 0, 1, 3, 0, 70.0, 1, 60.0, 1.2, 95.0, "Compliant", 5, 41.7, "Medium Risk",
                "Stable vocational progress.",
                0, None, 0, now, now
            ),
            (
                "DC-0920", "2026-CR-10119", "Tyler Jenkins", 39, "Male", "White", "High School Grad", 1,
                "Repeat Drug Possession & Trespass", "Phase I", "Level 3", "2026-06-22",
                "Hon. Evelyn Carter", "David Chen, MSW", "Officer Frank Miller",
                5, 2, 2, 2, 2, 2, 22.0, 4, 20.0, 0.4, 60.0, "Sanction Pending", 8, 75.4, "High Risk",
                "Missed 2 weekly drug screens.",
                0, None, 0, now, now
            )
        ]

        cursor.executemany("""
        INSERT INTO participants (
            id, docket_number, full_name, age, gender, race, education_level, dependents,
            offense_type, program_phase, supervision_level, enrollment_date,
            assigned_judge, assigned_case_manager, assigned_probation_officer,
            prior_felony_arrests, prior_misd_arrests, prior_violent_arrests, prior_property_arrests, prior_drug_arrests, prior_revocations,
            attendance_pct, unexcused_absences, employment_pct, jobs_per_year, drug_screening_compliance, treatment_status, compas_score,
            risk_probability, risk_category, notes, imported_from_nij, nij_record_id, is_archived, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, seed_participants)
        conn.commit()
        print("[✓] Seeded default court participants.")

    # Seed Initial Treatment Plans
    cursor.execute("SELECT COUNT(*) FROM treatment_plans;")
    if cursor.fetchone()[0] == 0:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        seed_plans = [
            ("DC-0812", "CM-4019-CLIN", "Phase II Intensive Recovery & Vocational Integration", "Complete 90 days dual-diagnosis therapy, maintain >85% attendance, secure vocational training", "CBT Individual Counseling, Group Relapse Prevention, Welding Vocational Program", "Twice Weekly", 85.0, 65.0, "Active", now),
            ("DC-0749", "CM-4019-CLIN", "Phase I Stabilization & Crisis Intervention", "Achieve 30 consecutive days clean screens, attend daily outpatient group", "Daily Day-Reporting Center, Individual Substance Assessment", "Daily", 90.0, 20.0, "Active", now),
            ("DC-0891", "CM-4019-CLIN", "Phase II Cognitive Restructuring & Childcare Support", "Address anxiety-driven relapse triggers, secure reliable childcare for clinical hours", "Trauma-Informed CBT, Parenting Support Group", "Weekly", 80.0, 50.0, "Active", now),
            ("DC-0511", "CM-4019-CLIN", "Phase III Community Re-integration & Peer Leadership", "Complete community service, mentor Phase I participants, full-time employment", "Peer Support Group, Bi-Weekly Clinical Check-in", "Bi-Weekly", 95.0, 90.0, "Active", now)
        ]
        cursor.executemany("""
        INSERT INTO treatment_plans (participant_id, case_manager_id, plan_name, clinical_goals, prescribed_interventions, counseling_frequency, target_attendance, target_employment, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, seed_plans)
        conn.commit()

    # Seed Initial Counseling Sessions
    cursor.execute("SELECT COUNT(*) FROM counseling_sessions;")
    if cursor.fetchone()[0] == 0:
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        seed_sessions = [
            ("DC-0812", "CM-4019-CLIN", "2026-11-04", "Individual CBT", "Attended", "Discussed stress triggers related to job loss. Participant receptive to vocational welding program referral.", 4),
            ("DC-0812", "CM-4019-CLIN", "2026-10-28", "Group Relapse Prevention", "Attended", "Active participation in peer group. Acknowledged recent attendance lapses.", 3),
            ("DC-0749", "CM-4019-CLIN", "2026-11-02", "Intake Assessment", "Attended", "Comprehensive substance history review. Identified severe housing instability.", 2),
            ("DC-0891", "CM-4019-CLIN", "2026-11-03", "Individual Therapy", "Attended", "Addressed prescription misuse triggers. Client obtained transportation voucher.", 4)
        ]
        cursor.executemany("""
        INSERT INTO counseling_sessions (participant_id, case_manager_id, session_date, session_type, attendance_status, clinical_notes, progress_rating)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, seed_sessions)
        conn.commit()

    # Seed Initial Drug Screenings
    cursor.execute("SELECT COUNT(*) FROM drug_screenings;")
    if cursor.fetchone()[0] == 0:
        seed_tests = [
            ("DC-0812", "PO-2281-DIST", "2026-11-03", "10-Panel Rapid Urinalysis", "Positive", "THC (50 ng/mL)", 1),
            ("DC-0812", "PO-2281-DIST", "2026-10-25", "10-Panel Rapid Urinalysis", "Negative", "None", 0),
            ("DC-0749", "PO-2281-DIST", "2026-11-01", "10-Panel Rapid Urinalysis", "Positive", "Cocaine Metabolite", 1),
            ("DC-0891", "PO-2281-DIST", "2026-10-29", "Lab GC/MS", "Negative", "None", 0),
            ("DC-0511", "PO-2281-DIST", "2026-10-28", "10-Panel Rapid Urinalysis", "Negative", "None", 0),
            ("DC-0422", "PO-2281-DIST", "2026-10-26", "10-Panel Rapid Urinalysis", "Negative", "None", 0)
        ]
        cursor.executemany("""
        INSERT INTO drug_screenings (participant_id, officer_id, screening_date, test_type, result, detected_substances, sanction_flag)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, seed_tests)
        conn.commit()

    # Seed Initial Violations
    cursor.execute("SELECT COUNT(*) FROM probation_violations;")
    if cursor.fetchone()[0] == 0:
        seed_violations = [
            ("DC-0812", "PO-2281-DIST", "2026-11-03", "Positive Drug Screen", "Moderate", "Tested positive for THC; attendance dropped to 12% in 30 days.", 1, "Reported"),
            ("DC-0749", "PO-2281-DIST", "2026-11-02", "Curfew Breach & Missed Attendance", "Critical", "GPS monitor indicates client breached 21:00 curfew by 3 hours; missed mandatory counseling session.", 1, "Hearing Scheduled"),
            ("DC-0920", "PO-2281-DIST", "2026-11-01", "Failure to Report", "Moderate", "Failed to appear for scheduled drug screen on two consecutive dates.", 1, "Reported")
        ]
        cursor.executemany("""
        INSERT INTO probation_violations (participant_id, officer_id, violation_date, violation_type, severity, description, reported_to_judge, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, seed_violations)
        conn.commit()

    # Seed Initial Field Visits
    cursor.execute("SELECT COUNT(*) FROM field_visits;")
    if cursor.fetchone()[0] == 0:
        seed_visits = [
            ("DC-0812", "PO-2281-DIST", "2026-11-08", "Home Curfew Verification", "Scheduled", "Verify residence stability and compliance with evening hours."),
            ("DC-0749", "PO-2281-DIST", "2026-11-07", "Unannounced Workplace Check", "Scheduled", "Inspect employer records and confirm attendance on shift."),
            ("DC-0664", "PO-2281-DIST", "2026-11-02", "Home Visit", "Completed", "Participant present. Residence clean and compliant.")
        ]
        cursor.executemany("""
        INSERT INTO field_visits (participant_id, officer_id, scheduled_date, visit_type, status, notes)
        VALUES (?, ?, ?, ?, ?, ?);
        """, seed_visits)
        conn.commit()

    # Seed Initial Notifications
    cursor.execute("SELECT COUNT(*) FROM notifications;")
    if cursor.fetchone()[0] == 0:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        seed_notifs = [
            ("Judge", "Sanction Hearing Docket Ready", "Marcus Vance (#DC-0812) and Jamal Thomas (#DC-0749) scheduled for 48-hr hearing.", "urgent", 0, now),
            ("Case Manager", "New Treatment Plan Milestone", "Elena Rostova (#DC-0511) has achieved 90-day attendance goal.", "info", 0, now),
            ("Probation Officer", "Positive Screening Alert", "Positive urinalysis logged for Marcus Vance (#DC-0812). Action required.", "warning", 0, now),
            ("Administrator", "System Calibration Complete", "All 5 ML architectures verified and calibrated. Accuracy: 83.5% Stacking.", "info", 0, now)
        ]
        cursor.executemany("""
        INSERT INTO notifications (user_role, title, message, type, is_read, created_at)
        VALUES (?, ?, ?, ?, ?, ?);
        """, seed_notifs)
        conn.commit()

    conn.close()
    print("[✓] SQLite Database initialized at:", DB_PATH)

if __name__ == "__main__":
    init_db()

