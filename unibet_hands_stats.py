# unibet_app.py
import json
import re
import sqlite3
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict
from typing import Optional

import streamlit as st
import pandas as pd
import altair as alt


# ----------------------- Hjälpfunktioner ----------------------- #

def parse_big_blind_eur(label: str):
    """Extraherar BB-storlek i € från t.ex. '€100 NL' → 1.0, '€50 NL' → 0.5."""
    if not isinstance(label, str):
        return None
    m = re.search(r"€\s*([0-9]+)", label)
    if not m:
        return None
    euros = int(m.group(1))
    return euros / 100.0


def fix_encoding(s: str) -> str:
    """Fixar enkla encoding-problem (latin1→utf8)."""
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin1").decode("utf-8")
    except Exception:
        return s


def format_cards_pretty_html(cards: str) -> str:
    """
    Gör om t.ex. 'kskd' → 'K♠ K♦' med färger via HTML:
      s → svart ♠, h → röd ♥, d → blå ♦, c → grön ♣
    """
    if not cards:
        return ""
    s = cards.strip().lower()
    suit_map = {"s": ("♠", "black"), "h": ("♥", "red"), "d": ("♦", "blue"), "c": ("♣", "green")}
    rank_map = {"t": "T", "j": "J", "q": "Q", "k": "K", "a": "A"}
    chunks = [s[i:i + 2] for i in range(0, len(s), 2)]
    pieces = []
    for ch in chunks:
        if len(ch) != 2:
            continue
        r_raw, su_raw = ch[0], ch[1]
        r = rank_map.get(r_raw, r_raw.upper())
        sym, color = suit_map.get(su_raw, ("?", "black"))
        span = f'<span style="color:{color}">{sym}</span>'
        pieces.append(f"{r}{span}")
    return " ".join(pieces)


def load_records_from_har_bytes(har_bytes: bytes, override_date: Optional[date] = None):
    """
    Läser händer ur HAR -> returnerar lista med records.
    Datum prioritering: override_date -> stime+1d -> startedDateTime.
    """
    har = json.loads(har_bytes.decode("utf-8"))
    entries = har.get("log", {}).get("entries", [])
    records = []
    level_labels = {}

    for entry in entries:
        req = entry.get("request", {})
        if "gethands" not in req.get("url", ""):
            continue

        stime_date = None
        body_text = req.get("postData", {}).get("text")
        if body_text:
            try:
                body = json.loads(body_text)
                stime = body.get("stime")
                if stime:
                    dt_stime = datetime.utcfromtimestamp(stime) + timedelta(days=1)
                    stime_date = dt_stime.date()
            except Exception:
                pass

        started_date = None
        started = entry.get("startedDateTime")
        if started:
            try:
                dt_started = datetime.fromisoformat(started.replace("Z", "+00:00"))
                started_date = dt_started.date()
            except Exception:
                pass

        if override_date is not None:
            the_date = override_date
        elif stime_date is not None:
            the_date = stime_date
        else:
            the_date = started_date

        if the_date:
            day_str = the_date.strftime("%Y-%m-%d")
            month_str = the_date.strftime("%Y-%m")
        else:
            day_str = None
            month_str = None

        content = entry.get("response", {}).get("content", {}) or {}
        text = content.get("text") or ""
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue

        refs = data.get("refs", {}) or {}
        for k, v in refs.items():
            try:
                level_labels[int(k)] = v
            except Exception:
                pass

        for h in data.get("hands", []):
            if len(h) < 9:
                continue
            hand_id = h[0]
            saldo_before = h[1]
            pot_cent = h[3]
            level_id = h[5]
            table_id = h[6]
            cards = h[7]
            saldo_after = h[8]

            raw_delta = saldo_after - saldo_before
            result_eur = -raw_delta / 100.0
            pot_eur = pot_cent / 100.0
            stake_label = refs.get(str(level_id), level_labels.get(level_id, f"Level {level_id}"))

            records.append({
                "hand_id": hand_id,
                "date": day_str,
                "month": month_str,
                "stake": stake_label,
                "table_id": table_id,
                "cards": cards,
                "pot_eur": pot_eur,
                "result_eur": result_eur,
            })

    return records


# ----------------------- SQLite-hantering ----------------------- #

def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection):
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hands (
                hand_id    INTEGER PRIMARY KEY,
                date       TEXT,
                month      TEXT,
                stake      TEXT,
                table_id   INTEGER,
                cards      TEXT,
                pot_eur    REAL,
                result_eur REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hands_date ON hands(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hands_month ON hands(month)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hands_stake ON hands(stake)")


def insert_hands(conn: sqlite3.Connection, records):
    if not records:
        return 0
    with conn:
        before = conn.total_changes
        conn.executemany("""
            INSERT OR IGNORE INTO hands
            (hand_id, date, month, stake, table_id, cards, pot_eur, result_eur)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (r["hand_id"], r["date"], r["month"], r["stake"], r["table_id"], r["cards"], r["pot_eur"], r["result_eur"])
            for r in records
        ])
        after = conn.total_changes
    return after - before


@st.cache_data
def load_all_hands_cached(db_path_str: str, db_mtime: float):
    conn = sqlite3.connect(db_path_str)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT hand_id, date, month, stake, table_id, cards, pot_eur, result_eur FROM hands"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# -------------------- Rake-estimat funktioner (parametriserade) -------------------- #

def get_rake_bb100_for_label(label: str, nl_default: float, plo_default: float) -> float:
    """Returnerar vilken BB/100 rake som bör användas för en stake-label."""
    if not isinstance(label, str):
        return nl_default
    s = label.lower()
    if "pl" in s or "plo" in s:
        return plo_default
    return nl_default


def estimate_rake_per_stake_summary(records, nl_rake_bb100: float, plo_rake_bb100: float, rakeback_pct: float):
    """
    Summerar per stake:
      - hands
      - result_eur
      - est_rake_eur
      - est_rakeback_eur
      - est_rakeback_bb100 (rake_bb100 * rakeback_pct)
    Returnerar dict stake->stats och totals.
    """
    per = {}
    total_result = 0.0
    total_rake = 0.0
    for r in records:
        stake = r.get("stake")
        if stake not in per:
            per[stake] = {"hands": 0, "result_eur": 0.0, "est_rake_eur": 0.0}
        per[stake]["hands"] += 1
        per[stake]["result_eur"] += r.get("result_eur", 0.0)

        bb_size = parse_big_blind_eur(stake) or 1.0
        rake_bb100 = get_rake_bb100_for_label(stake, nl_rake_bb100, plo_rake_bb100)
        # rake per hand in euro
        rake_per_hand_eur = (rake_bb100 / 100.0) * bb_size
        per[stake]["est_rake_eur"] += rake_per_hand_eur

        total_result += r.get("result_eur", 0.0)
        total_rake += rake_per_hand_eur

    # compute per-stake additional fields
    per_out = {}
    for stake, v in per.items():
        rake_bb100 = get_rake_bb100_for_label(stake, nl_rake_bb100, plo_rake_bb100)
        est_rakeback_bb100 = rake_bb100 * rakeback_pct
        est_rakeback_eur = v["est_rake_eur"] * rakeback_pct
        per_out[stake] = {
            "hands": v["hands"],
            "result_eur": v["result_eur"],
            "est_rake_eur": v["est_rake_eur"],
            "est_rakeback_eur": est_rakeback_eur,
            "est_rakeback_bb100": est_rakeback_bb100,
            "rake_bb100": rake_bb100,
        }

    totals = {
        "total_result_eur": total_result,
        "total_est_rake_eur": total_rake,
        "total_est_rakeback_eur": total_rake * rakeback_pct
    }

    return per_out, totals


# ----------------------- Aggregat-funktioner ----------------------- #

def aggregate_hands(records):
    if not records:
        return {"hands": 0, "result_eur": 0.0, "bb100": 0.0, "stakes": {}}
    stakes = defaultdict(lambda: {"hands": 0, "eur": 0.0, "bb_sum": 0.0})
    total_hands = 0
    total_eur = 0.0
    total_bb_sum = 0.0
    for r in records:
        eur = r["result_eur"]
        stake = r["stake"]
        bb_size = parse_big_blind_eur(stake) or 1.0
        bb = eur / bb_size
        stakes[stake]["hands"] += 1
        stakes[stake]["eur"] += eur
        stakes[stake]["bb_sum"] += bb
        total_hands += 1
        total_eur += eur
        total_bb_sum += bb
    bb100 = total_bb_sum / (total_hands / 100.0) if total_hands > 0 else 0.0
    stakes_out = {}
    for stake, d in stakes.items():
        h = d["hands"]
        stakes_out[stake] = {
            "hands": h,
            "result_eur": d["eur"],
            "bb100": d["bb_sum"] / (h / 100.0) if h > 0 else 0.0,
        }
    return {"hands": total_hands, "result_eur": total_eur, "bb100": bb100, "stakes": stakes_out}


def aggregate_by_date(all_hands):
    by_date = defaultdict(list)
    for r in all_hands:
        if r.get("date"):
            by_date[r["date"]].append(r)
    return {d: aggregate_hands(recs) for d, recs in by_date.items()}


def aggregate_by_month(all_hands):
    by_month = defaultdict(list)
    for r in all_hands:
        if r.get("month"):
            by_month[r["month"]].append(r)
    return {m: aggregate_hands(recs) for m, recs in by_month.items()}


# ------------------------ STREAMLIT GUI ------------------------ #

def main():
    st.title("Unibet HAR → SQLite-handdatabas & resultat")

    default_folder = Path("unibet_data").resolve()
    folder_str = st.sidebar.text_input("Datamapp", str(default_folder))
    data_folder = Path(folder_str)
    data_folder.mkdir(parents=True, exist_ok=True)
    db_path = data_folder / "unibet_hands.sqlite"
    st.sidebar.write(f"Databasfil: `{db_path}`")

    # --- Rake-parametrar i sidopanel ---
    st.sidebar.header("Rake-inställningar (BB/100)")
    nl_rake_bb100 = st.sidebar.number_input("NL rake (BB/100)", value=8.0, min_value=0.0, step=0.5)
    plo_rake_bb100 = st.sidebar.number_input("PLO rake (BB/100)", value=12.0, min_value=0.0, step=0.5)
    rakeback_pct = st.sidebar.slider("Rakeback %", min_value=0, max_value=100, value=40, step=1) / 100.0
    st.sidebar.caption("Skriv in dina antaganden här. Dessa används för estimering per stake.")

    conn = get_connection(db_path)
    init_db(conn)

    st.sidebar.header("Importera HAR")
    use_manual_date = st.sidebar.checkbox("Sätt datum manuellt")
    manual_date: Optional[date] = None
    if use_manual_date:
        manual_date = st.sidebar.date_input("Datum för dessa HAR", value=date.today())

    uploads = st.sidebar.file_uploader("Välj en eller flera HAR-filer", type=["har"], accept_multiple_files=True)
    if uploads:
        total_added = 0
        for uf in uploads:
            st.sidebar.write(f"Läser {uf.name}...")
            recs = load_records_from_har_bytes(uf.getvalue(), override_date=manual_date if use_manual_date else None)
            added = insert_hands(conn, recs)
            total_added += added
            st.sidebar.success(f"{uf.name}: {added} nya händer")
        st.sidebar.info(f"Totalt nya händer denna import: {total_added}")

    if db_path.exists():
        db_mtime = db_path.stat().st_mtime
    else:
        db_mtime = 0.0

    all_hands = load_all_hands_cached(str(db_path), db_mtime)
    if not all_hands:
        st.info("Inga händer i databasen ännu. Importera en HAR-fil.")
        return

    st.sidebar.write(f"Totalt antal händer: **{len(all_hands)}**")

    view = st.radio("Vy", ["Per dag", "Per månad", "Alla dagar (tabell + grafer)", "Alla händer (sortable)",
                          "Graf – kumulativ resultatkurva", "Total (alla händer)"])

    # --- PER DAG ---
    if view == "Per dag":
        per_date = aggregate_by_date(all_hands)
        dates = sorted(per_date.keys(), reverse=True) # Senaste först
        chosen = st.selectbox("Datum", dates)
        agg = per_date[chosen]
        st.header(f"Datum: {chosen}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Resultat (€)", f"{agg['result_eur']:.2f}")
        col2.metric("Händer", agg["hands"])
        col3.metric("BB/100", f"{agg['bb100']:.2f}")
        rows = []
        for stake, info in agg["stakes"].items():
            rows.append({"Stake": fix_encoding(stake), "Händer": info["hands"],
                         "Result (€)": round(info["result_eur"], 2), "BB/100": round(info["bb100"], 2)})
        st.subheader("Per stake")
        st.table(pd.DataFrame(rows))

    # --- PER MÅNAD ---
    elif view == "Per månad":
        per_month = aggregate_by_month(all_hands)
        months = sorted(per_month.keys())
        chosen = st.selectbox("Månad", months)
        agg = per_month[chosen]
        st.header(f"Månad: {chosen}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Resultat (€)", f"{agg['result_eur']:.2f}")
        col2.metric("Händer", agg["hands"])
        col3.metric("BB/100", f"{agg['bb100']:.2f}")
        rows = []
        for stake, info in agg["stakes"].items():
            rows.append({"Stake": fix_encoding(stake), "Händer": info["hands"],
                         "Result (€)": round(info["result_eur"], 2), "BB/100": round(info["bb100"], 2)})
        st.subheader("Per stake")
        st.table(pd.DataFrame(rows))

    # --- ALLA DAGAR (TABELL + GRAF) ---
    elif view == "Alla dagar (tabell + grafer)":
        per_date = aggregate_by_date(all_hands)
        rows = []
        for d, agg in per_date.items():
            rows.append({"Datum": d, "Händer": agg["hands"], "Result (€)": round(agg["result_eur"], 2),
                         "BB/100": round(agg["bb100"], 2)})
        df = pd.DataFrame(rows).sort_values("Datum")
        st.subheader("Tabell per dag")
        st.table(df)
        df_plot = df.copy()
        df_plot["Datum"] = pd.to_datetime(df_plot["Datum"], errors="coerce")
        df_plot = df_plot.set_index("Datum").sort_index()
        if not df_plot.empty:
            st.subheader("Resultat (€) över dagar")
            st.line_chart(df_plot["Result (€)"])
            st.subheader("BB/100 över dagar")
            st.line_chart(df_plot["BB/100"])

    # --- ALLA HÄNDER (FILTER + LIMIT) ---
    elif view == "Alla händer (sortable)":
        st.header("Alla händer – sortera / filtrera")
        df = pd.DataFrame(all_hands)
        if df.empty:
            st.info("Inga händer att visa.")
        else:
            df["Datum"] = pd.to_datetime(df["date"], errors="coerce")
            df["Stake_label"] = df["stake"].apply(fix_encoding)
            col_f1, col_f2 = st.columns(2)
            stakes_all = sorted(df["Stake_label"].dropna().unique().tolist())
            with col_f1:
                selected_stakes = st.multiselect("Stakefilter", stakes_all, default=stakes_all)
            min_d = df["Datum"].min()
            max_d = df["Datum"].max()
            with col_f2:
                if pd.isna(min_d) or pd.isna(max_d):
                    date_range = (None, None)
                else:
                    date_range = st.date_input("Datumintervall", value=(min_d.date(), max_d.date()),
                                               min_value=min_d.date(), max_value=max_d.date())
            col_f3, col_f4 = st.columns(2)
            with col_f3:
                sort_col = st.selectbox("Sortera på", ["Pott (€)", "Result (€)", "Datum", "Hand ID"], index=0)
            with col_f4:
                max_rows = st.slider("Max antal händer att visa", 100, 5000, 2000, step=100)
            df_f = df.copy()
            if selected_stakes:
                df_f = df_f[df_f["Stake_label"].isin(selected_stakes)]
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_d, end_d = date_range
                if start_d is not None and end_d is not None:
                    mask = (df_f["Datum"].dt.date >= start_d) & (df_f["Datum"].dt.date <= end_d)
                    df_f = df_f[mask]
            if df_f.empty:
                st.info("Inga händer matchar filtren.")
            else:
                df_f["BB_size"] = df_f["stake"].apply(lambda s: parse_big_blind_eur(s) or 1.0)
                df_f["Result (€)"] = df_f["result_eur"]
                df_f["BB"] = df_f["Result (€)"] / df_f["BB_size"]
                df_f["Pott (€)"] = df_f["pot_eur"]
                df_f["Hand ID"] = df_f["hand_id"]
                df_f["Bord"] = df_f["table_id"]
                sort_map = {"Pott (€)": "Pott (€)", "Result (€)": "Result (€)", "Datum": "Datum", "Hand ID": "Hand ID"}
                sort_key = sort_map.get(sort_col, "Pott (€)")
                df_f = df_f.sort_values(sort_key, ascending=False)
                total_matches = len(df_f)
                df_view = df_f.head(max_rows).copy()
                df_view["Kort"] = df_view["cards"].apply(format_cards_pretty_html)
                df_view["Result (€)"] = df_view["Result (€)"].round(2)
                df_view["BB"] = df_view["BB"].round(2)
                df_view["Pott (€)"] = df_view["Pott (€)"].round(2)
                df_view["Datum_str"] = df_view["Datum"].dt.strftime("%Y-%m-%d %H:%M:%S")
                out_cols = ["Datum_str", "Hand ID", "Stake_label", "Bord", "Kort", "Pott (€)", "Result (€)", "BB"]
                df_out = df_view[out_cols].rename(columns={"Datum_str": "Datum", "Stake_label": "Stake"})
                st.caption(f"Visar {len(df_out)} av {total_matches} händer (filtrerade och begränsade).")
                html_table = df_out.to_html(escape=False, index=False)
                st.markdown(html_table, unsafe_allow_html=True)

    # --- KUMULATIV RESULTATKURVA (med stake-filter & ned-sampling + rakeback-line) ---
    elif view == "Graf – kumulativ resultatkurva":
        st.header("Kumulativ resultatkurva – med stake-filter och rakeback-linje")
        df = pd.DataFrame(all_hands)
        if df.empty:
            st.info("Inga händer att visa.")
        else:
            df["Datum"] = pd.to_datetime(df["date"], errors="coerce")
            stakes_all = sorted({fix_encoding(s) for s in df["stake"].unique()})
            df["stake_label"] = df["stake"].apply(fix_encoding)
            selected_stakes = st.multiselect("Välj vilka stakes som ska ingå i grafen", stakes_all, default=stakes_all)
            if not selected_stakes:
                st.info("Välj minst en stake för att se grafen.")
            else:
                df_sel = df[df["stake_label"].isin(selected_stakes)].copy()
                if df_sel.empty:
                    st.info("Inga händer för de valda stakesen.")
                else:
                    df_sel = df_sel.sort_values(["Datum", "hand_id"]).reset_index(drop=True)
                    # per-hand rake & rakeback (i ordning)
                    per_hand = []
                    for _, r in df_sel.iterrows():
                        stake_label = r["stake"]
                        bb_size = parse_big_blind_eur(stake_label) or 1.0
                        rake_bb100 = get_rake_bb100_for_label(stake_label, nl_rake_bb100, plo_rake_bb100)
                        rake_per_hand_eur = (rake_bb100 / 100.0) * bb_size
                        rakeback_per_hand_eur = rake_per_hand_eur * rakeback_pct
                        per_hand.append((rake_per_hand_eur, rakeback_per_hand_eur))
                    df_sel["rake_eur"] = [p[0] for p in per_hand]
                    df_sel["rakeback_eur"] = [p[1] for p in per_hand]
                    df_sel["eur"] = df_sel["result_eur"]
                    df_sel["cum_observed"] = df_sel["eur"].cumsum()
                    df_sel["cum_rakeback"] = df_sel["rakeback_eur"].cumsum()
                    df_sel["cum_with_rakeback"] = df_sel["cum_observed"] + df_sel["cum_rakeback"]
                    df_sel["hand_index"] = range(1, len(df_sel) + 1)

                    # ned-sampling
                    max_points = 5000
                    n = len(df_sel)
                    if n > max_points:
                        step = max(1, n // max_points)
                        df_plot = df_sel.iloc[::step].copy()
                    else:
                        df_plot = df_sel
                    df_plot = df_plot.set_index("hand_index")
                    st.caption(f"Visar {len(df_plot)} av {n} händer (ned-samplad graf).")

                    # skapa long-form för Altair
                    plot_df = pd.DataFrame({
                        "hand_index": df_plot.index,
                        "Observerat (EUR)": df_plot["cum_observed"].values,
                        "Med rakeback (EUR)": df_plot["cum_with_rakeback"].values
                    }).melt(id_vars="hand_index", var_name="serie", value_name="cum_eur")

                    # altair chart med färger och legend
                    color_scale = alt.Scale(domain=["Observerat (EUR)", "Med rakeback (EUR)"],
                                            range=["#7ec8ff", "#0066cc"])
                    chart = alt.Chart(plot_df).mark_line().encode(
                        x="hand_index:Q",
                        y="cum_eur:Q",
                        color=alt.Color("serie:N", scale=color_scale, legend=alt.Legend(title="Serie")),
                    ).properties(width=800, height=350)
                    st.altair_chart(chart, use_container_width=True)

                    # BB-version (valfritt)
                    if st.checkbox("Visa kumulativ BB-version istället för EUR"):
                        df_sel["bb_size"] = df_sel["stake"].apply(lambda s: parse_big_blind_eur(s) or 1.0)
                        df_sel["bb_per_hand"] = df_sel["eur"] / df_sel["bb_size"]
                        df_sel["rakeback_bb_per_hand"] = df_sel["rakeback_eur"] / df_sel["bb_size"]
                        df_sel["cum_bb_observed"] = df_sel["bb_per_hand"].cumsum()
                        df_sel["cum_bb_with_rakeback"] = df_sel["cum_bb_observed"] + df_sel["rakeback_bb_per_hand"].cumsum()

                        if n > max_points:
                            df_plot_bb = df_sel.iloc[::step].copy()
                        else:
                            df_plot_bb = df_sel
                        df_plot_bb = df_plot_bb.set_index("hand_index")
                        plot_bb_df = pd.DataFrame({
                            "hand_index": df_plot_bb.index,
                            "Observerat (BB cumulative)": df_plot_bb["cum_bb_observed"].values,
                            "Med rakeback (BB cumulative)": df_plot_bb["cum_bb_with_rakeback"].values
                        }).melt(id_vars="hand_index", var_name="serie", value_name="cum_val")
                        color_scale_bb = alt.Scale(domain=["Observerat (BB cumulative)", "Med rakeback (BB cumulative)"],
                                                   range=["#7ec8ff", "#0066cc"])
                        chart_bb = alt.Chart(plot_bb_df).mark_line().encode(
                            x="hand_index:Q",
                            y="cum_val:Q",
                            color=alt.Color("serie:N", scale=color_scale_bb, legend=alt.Legend(title="Serie"))
                        ).properties(width=800, height=350)
                        st.altair_chart(chart_bb, use_container_width=True)

    # --- TOTAL (alla händer) med per-stake estimerad rake/rakeback integrerat ---
    else:
        agg = aggregate_hands(all_hands)
        st.header("Total – alla händer")
        # compute per-stake rake summary & totals
        per_stake_summary, totals = estimate_rake_per_stake_summary(all_hands, nl_rake_bb100, plo_rake_bb100, rakeback_pct)

        # top metrics: Resultat, Händer, BB/100, Resultat + rakeback
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Resultat (€)", f"{agg['result_eur']:.2f}")
        col2.metric("Händer", agg["hands"])
        col3.metric("BB/100", f"{agg['bb100']:.2f}")
        # totals may contain total_result_eur; if not, fall back to agg
        total_result = totals.get("total_result_eur", agg["result_eur"])
        total_est_rakeback = totals.get("total_est_rakeback_eur", 0.0)
        result_plus_rakeback = total_result + total_est_rakeback
        col4.metric("Resultat + rakeback (EUR)", f"{result_plus_rakeback:.2f}")

        # Per stake (resultat)
        rows = []
        for stake, info in agg["stakes"].items():
            rows.append({"Stake": fix_encoding(stake), "Händer": info["hands"],
                         "Result (€)": round(info["result_eur"], 2), "BB/100": round(info["bb100"], 2)})
        st.subheader("Per stake (resultat)")
        df_stakes = pd.DataFrame(rows)
        st.table(df_stakes)

        # Integrerad per-stake rake summary (i Total-vyn, en tabell)
        rows_r = []
        for stake_label, info in per_stake_summary.items():
            rows_r.append({
                "Stake": fix_encoding(stake_label),
                "Händer": info["hands"],
                "Result (EUR)": round(info["result_eur"], 2),
                "Est. rake (EUR)": round(info["est_rake_eur"], 2),
                "Est. rakeback (EUR)": round(info["est_rakeback_eur"], 2),
                "Rakeback (BB/100)": round(info["est_rakeback_bb100"], 3),
            })
        if rows_r:
            st.subheader("Per stake: estimerad rake & rakeback (integrerat)")
            st.table(pd.DataFrame(rows_r))

        # summary totals (observerat + totals)
        st.subheader("Totalsammanfattning")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Observerat resultat (EUR)", f"{total_result:.2f}")
        col_b.metric("Est. rake (EUR)", f"{totals['total_est_rake_eur']:.2f}")
        col_c.metric("Est. rakeback (EUR)", f"{totals['total_est_rakeback_eur']:.2f}")

        st.caption("Per-stake-tabellen visar estimerad rake och estimerad rakeback. "
                   "Rakeback (BB/100) = (rake BB/100) * rakeback% för respektive stake.")


if __name__ == "__main__":
    main()
