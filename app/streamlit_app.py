"""ChurnLens Streamlit app.

Run:  streamlit run app/streamlit_app.py
Reads only pipeline outputs and the read-only tool layer; nothing here writes data or
sends messages. Every page states the data mode and sample it comes from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from churnlens.agents import supervisor
from churnlens.analysis.plotting import SERIES
from churnlens.config import load_config, resolve_path
from churnlens.explain.run import plot_waterfall
from churnlens.tools import core

st.set_page_config(page_title="ChurnLens", layout="wide")
DATA = load_config("data")
PCT = 100 if DATA["mode"] == "full" else DATA["user_sample_pct"]
PROVENANCE = (f"Data: KKBox, {DATA['mode']} mode, {PCT}% user sample (seed {DATA['sample_seed']}). "
              "Associations, not causes. Costs, save rates and margins are placeholders.")
PAGES = ["Executive KPIs", "Risk × value", "Segments", "Drivers", "Customer lookup",
         "Budget simulator", "Agent chat", "Live scoring (April)"]


def path(key: str) -> Path:
    return resolve_path(key, DATA)


@st.cache_data
def sql(query: str) -> pd.DataFrame:
    res = core.run_sql(query, max_rows=100000)
    return pd.DataFrame(res["rows"], columns=res["columns"])


@st.cache_data
def csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(path(rel))


@st.cache_data
def model_summary() -> dict:
    return json.loads(path("reports/model/summary.json").read_text(encoding="utf-8"))


def page_kpis() -> None:
    st.header("Executive KPIs")
    s = model_summary()
    test = s["models"][s["selected"]]["test"]
    churn = sql("SELECT AVG(is_churn) AS a FROM labels_a").iloc[0, 0]
    churn_b = sql("SELECT AVG(is_churn) AS b FROM labels_b").iloc[0, 0]
    q = sql("SELECT risk_band, COUNT(*) AS n, SUM(evar) AS evar FROM customers GROUP BY 1")
    high = q.set_index("risk_band")
    c = st.columns(5)
    c[0].metric("Churn rate, snapshot A (train)", f"{churn:.2%}")
    c[1].metric("Churn rate, snapshot B (test)", f"{churn_b:.2%}")
    c[2].metric("Model PR-AUC on B", f"{test['pr_auc']:.3f}", help="Base rate = B churn rate")
    c[3].metric("High-risk customers (B)", f"{int(high.loc['high', 'n']):,}")
    c[4].metric("Value at risk, high-risk band", f"{high.loc['high', 'evar']:,.0f} NTD")
    pol = csv("reports/prioritize/B_policies.csv")
    st.subheader("Contact-list policies (budget and capacity from configs/prioritize.yaml)")
    st.dataframe(pol, hide_index=True)
    top = csv("reports/model/top_k.csv")
    st.subheader("Ranking quality on B: contact the top K%")
    st.dataframe(top, hide_index=True)


def page_risk_value() -> None:
    st.header("Risk × value matrix (snapshot B)")
    df = sql("SELECT p_churn, cltv, evar, quadrant, selected FROM customers USING SAMPLE 15000 "
             "ROWS (reservoir, 42)")
    fig = px.scatter(df, x="cltv", y="p_churn", color="quadrant", log_y=True, opacity=0.5,
                     color_discrete_sequence=SERIES,
                     labels={"cltv": "CLTV (NTD)", "p_churn": "P(churn), log scale"})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(sql("SELECT quadrant, COUNT(*) AS customers, AVG(p_churn) AS mean_p, "
                     "AVG(cltv) AS mean_cltv, SUM(evar) AS total_evar, SUM(selected::INT) AS "
                     "on_contact_list FROM customers GROUP BY 1 ORDER BY 1"), hide_index=True)


def page_segments() -> None:
    st.header("Segments")
    seg = csv("reports/rfm/segment_rates.csv")
    st.subheader("RFM segments: churn rate in snapshot A (rule order)")
    st.plotly_chart(px.bar(seg, x="segment", y="rate", color_discrete_sequence=SERIES,
                           labels={"rate": "churn rate"}), width="stretch")
    st.dataframe(seg, hide_index=True)
    st.subheader("Customer groups (K-Means on payers who listened, + rule labels)")
    st.dataframe(csv("reports/rfm/cluster_profiles.csv"), hide_index=True)


def page_drivers() -> None:
    st.header("Drivers (SHAP, what the model relies on — not causes)")
    g = csv("reports/explain/B_group_importance.csv")
    g.columns = ["group", "mean_abs_shap"]
    st.plotly_chart(px.bar(g, x="mean_abs_shap", y="group", orientation="h",
                           color_discrete_sequence=SERIES), width="stretch")
    st.subheader("Most common #1 driver")
    st.dataframe(sql("SELECT cluster, COUNT(*) AS customers FROM drivers WHERE rank = 1 "
                     "GROUP BY 1 ORDER BY 2 DESC LIMIT 15"), hide_index=True)
    st.image(str(path("reports/figures/phase7_shap_summary.png")))


def page_lookup() -> None:
    st.header("Customer lookup")
    top = sql("SELECT msno FROM customers WHERE selected ORDER BY priority_rank LIMIT 50")
    choice = st.selectbox("Pick from the contact list", top["msno"].tolist())
    msno = st.text_input("…or paste a customer id (msno)", value=choice).strip()
    try:
        risk = core.get_risk_and_drivers(msno)
        value = core.get_cltv(msno)
        act = core.recommend_action(msno)
        prof = core.get_customer_profile(msno)
    except Exception as e:  # unknown id or bad input: show the reason
        st.error(str(e))
        return
    c = st.columns(4)
    c[0].metric("P(churn), calibrated", f"{risk['p_churn']:.1%}")
    c[1].metric("CLTV if retained", f"{value['cltv_used']:,.0f} NTD")
    c[2].metric("Value at risk", f"{value['expected_value_at_risk']:,.0f} NTD")
    c[3].metric("Recommended action", act["action"])
    st.write("**Top drivers** (associations, not causes):")
    for d in risk["drivers"]:
        st.write(f"{d['rank']}. {d['reason']}")
    shap = pd.read_parquet(path("data/processed/{mode}/explain/B_shap.parquet"),
                           filters=[("msno", "==", msno)])
    meta = json.loads(path("data/processed/{mode}/explain/B_shap_meta.json")
                      .read_text(encoding="utf-8"))
    if len(shap):
        row = shap.iloc[0]
        contrib = row.drop(["msno", "raw_score", "p_churn"]).astype(float)
        fig_path = path("reports/agent_charts/waterfall_lookup.png")
        plot_waterfall(contrib, meta["base_value"], float(row["raw_score"]),
                       float(row["p_churn"]), "lookup", 10, PROVENANCE, fig_path, 110)
        st.image(str(fig_path))
    with st.expander("Profile facts"):
        st.json(prof["profile"])


def page_budget() -> None:
    st.header("Budget simulator")
    budget = st.slider("Budget (NTD)", 0, 500_000, 150_000, step=10_000)
    cap = st.slider("Outreach capacity (contacts)", 0, 10_000, 4_000, step=250)
    res = core.optimize_budget(float(budget), int(cap), top_n=25)
    c = st.columns(4)
    c[0].metric("Customers contacted", f"{res['contacted']:,}")
    c[1].metric("Spend", f"{res['spend']:,.0f} NTD")
    c[2].metric("Expected net benefit", f"{res['expected_net_benefit']:,.0f} NTD")
    c[3].metric("Upper bound", f"{res['upper_bound']:,.0f} NTD")
    mix = pd.DataFrame([{"action": a, **v} for a, v in res["action_mix"].items()])
    if not mix.empty:
        st.plotly_chart(px.bar(mix, x="action", y="users", color_discrete_sequence=SERIES),
                        width="stretch")
    st.dataframe(pd.DataFrame(res["top"]), hide_index=True)
    st.caption(res["note"])


def page_chat() -> None:
    st.header("Agent chat")
    st.caption("Offline agents unless ANTHROPIC_API_KEY and CHURNLENS_MODEL are set. Read-only; "
               "drafts are never sent.")
    if "history" not in st.session_state:
        st.session_state.history = []
    for role, text in st.session_state.history:
        st.chat_message(role).write(text)
    q = st.chat_input("Ask about churn, a customer id, a budget, or drift…")
    if q:
        st.session_state.history.append(("user", q))
        st.chat_message("user").write(q)
        r = supervisor.ask(q)
        reply = f"_{r.agent} agent · {r.mode} · {r.latency_s:.2f}s_\n\n{r.answer}"
        st.session_state.history.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)
            for c in r.charts:
                st.image(c)


def page_live() -> None:
    st.header("Live scoring: April 2017 expirations (snapshot C)")
    f = path("data/processed/{mode}/scoring/C_scores.parquet")
    if not f.exists():
        st.warning("Run `python -m churnlens.scoring.live` first.")
        return
    df = pd.read_parquet(f)
    c = st.columns(3)
    c[0].metric("Users scored", f"{len(df):,}")
    c[1].metric("Mean P(churn)", f"{df['p_churn'].mean():.2%}")
    c[2].metric("High risk", f"{(df['risk_band'] == 'high').mean():.1%}")
    st.plotly_chart(px.histogram(df, x="p_churn", nbins=50, log_y=True,
                                 color_discrete_sequence=SERIES), width="stretch")
    st.dataframe(df.sort_values("evar", ascending=False).head(25), hide_index=True)
    st.caption("No labels exist yet for April; these are predictions only. Uses the "
               "scoring-only March logs, which never feed training.")


def main() -> None:
    st.sidebar.title("ChurnLens")
    page = st.sidebar.radio("Page", PAGES)
    st.sidebar.caption(PROVENANCE)
    {"Executive KPIs": page_kpis, "Risk × value": page_risk_value, "Segments": page_segments,
     "Drivers": page_drivers, "Customer lookup": page_lookup, "Budget simulator": page_budget,
     "Agent chat": page_chat, "Live scoring (April)": page_live}[page]()


main()
