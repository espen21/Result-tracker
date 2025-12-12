import sqlite3
from pathlib import Path

OLD_DB = Path("Old_db/unibet_hands_old.sqlite")   # ändra
NEW_DB = Path("unibet_data/unibet_hands.sqlite")   # ändra



TABLE = "hands"

def migrate_missing_rows_strict():
    conn = sqlite3.connect(NEW_DB)
    conn.execute(f"ATTACH DATABASE '{OLD_DB}' AS old")

    # Rader i gamla med ogiltigt datum (som skulle ge unix_date NULL)
    bad_dates = conn.execute(f"""
        SELECT COUNT(*)
        FROM old.{TABLE}
        WHERE date IS NULL OR strftime('%s', date) IS NULL
    """).fetchone()[0]

    # För över bara saknade hand_id + bara giltiga dates
    conn.execute(f"""
        INSERT INTO {TABLE} (
            hand_id, date, month, stake, unix_date, cards, pot_eur, result_eur
        )
        SELECT
            o.hand_id,
            o.date,
            o.month,
            o.stake,
            CAST(strftime('%s', o.date) AS INTEGER) AS unix_date,
            o.cards,
            o.pot_eur,
            o.result_eur
        FROM old.{TABLE} o
        WHERE strftime('%s', o.date) IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM {TABLE} n WHERE n.hand_id = o.hand_id
          );
    """)

    conn.commit()
    inserted = conn.execute("SELECT changes()").fetchone()[0]

    conn.execute("DETACH DATABASE old")
    conn.close()

    print(f"✅ Infogade rader: {inserted}")
    print(f"⚠️ Rader med ogiltigt date i gamla DB (skippade): {bad_dates}")

if __name__ == "__main__":
    migrate_missing_rows_strict()
