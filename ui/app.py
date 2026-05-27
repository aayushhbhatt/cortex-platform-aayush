from __future__ import annotations

from pathlib import Path
import sys
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import os
from agents.supervisor import invoke_with_trace
from dotenv import load_dotenv
from evaluation.evaluate import evaluate_retrieval
from ui.helpers import (
    AGENT_DESCRIPTIONS,
    build_agent_capability_rows,
    build_agent_detail_rows,
    build_routing_explanation,
    build_run_summary,
    build_timeline_rows,
    citations_to_rows,
    get_agent_debug,
    get_router_debug,
    memory_summary,
    research_results_to_rows,
    rag_access_rows,
    rag_generation_rows,
    rag_query_processing_rows,
    rag_ranking_debug_rows,
    rag_retrieval_results_rows,
    get_knowledge_rag_debug,
    safe_jsonable,
    summarize_router_debug,
    tool_results_to_rows,
    truncate_text,
    used_chunks_to_rows,
)

load_dotenv()
st.set_page_config(page_title="Cortex — Agent Command Center", page_icon="🧠", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1320px;}
        [data-testid="stSidebar"] {background-color: #0f172a;}
        [data-testid="stSidebar"] * {color: #f8fafc;}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {color: #e5e7eb;}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {color: #ffffff; font-weight: 700;}
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            background-color: #ffffff;
            color: #111827;
            border: 1px solid #cbd5e1;
            font-weight: 500;
        }
        [data-testid="stSidebar"] input::placeholder,
        [data-testid="stSidebar"] textarea::placeholder {color: #6b7280; opacity: 1;}
        [data-testid="stSidebar"] input:disabled,
        [data-testid="stSidebar"] textarea:disabled {color: #374151; opacity: 1;}
        [data-testid="stSidebar"] [role="radiogroup"] label,
        [data-testid="stSidebar"] [role="radiogroup"] span {color: #f8fafc; font-weight: 500;}
        [data-testid="stSidebar"] [data-baseweb="select"] {background-color: #ffffff;}
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [data-baseweb="select"] div {color: #111827;}
        [data-testid="stSidebar"] [data-testid="stExpander"] {color: #f8fafc;}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] p {color: #e5e7eb;}
        [data-testid="stSidebar"] .sidebar-help,
        [data-testid="stSidebar"] .sidebar-caption {color: #cbd5e1; font-size: 0.9rem; line-height: 1.5;}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #cbd5e1;}
        [data-testid="stSidebar"] .stButton button {color: #ffffff; font-weight: 700;}
        .cortex-hero {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 24px;
            padding: 24px 28px;
            background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.94));
            color: white;
            margin-bottom: 18px;
            box-shadow: 0 14px 36px rgba(15,23,42,0.18);
        }
        .cortex-hero h1 {font-size: 2.15rem; margin: 0 0 0.35rem 0; letter-spacing: -0.03em;}
        .cortex-hero p {font-size: 1rem; color: #cbd5e1; margin: 0; max-width: 860px;}
        .soft-card {
            border: 1px solid rgba(148, 163, 184, 0.24);
            background: rgba(255,255,255,0.78);
            border-radius: 20px;
            padding: 18px 20px;
            box-shadow: 0 10px 28px rgba(15,23,42,0.08);
            margin-bottom: 12px;
        }
        .answer-card {
            border: 1px solid rgba(59,130,246,0.22);
            background: linear-gradient(180deg, rgba(239,246,255,0.98), rgba(255,255,255,0.98));
            border-radius: 22px;
            padding: 22px 24px;
            box-shadow: 0 12px 28px rgba(37,99,235,0.10);
            margin: 10px 0 18px 0;
        }
        .answer-card h3 {margin-top: 0; font-size: 1rem; color: #1e3a8a; text-transform: uppercase; letter-spacing: 0.08em;}
        .answer-card div {font-size: 1.05rem; line-height: 1.62; color: #0f172a;}
        .mini-label {font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.15rem;}
        .mini-value {font-size: 1.25rem; font-weight: 750; color: #0f172a;}
        .pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.35rem 0.72rem;
            margin: 0.14rem 0.22rem 0.14rem 0;
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            color: #3730a3;
            font-weight: 650;
            font-size: 0.84rem;
        }
        .pill-muted {background: #f8fafc; border-color: #e2e8f0; color: #334155;}
        .pill-ok {background: #ecfdf5; border-color: #bbf7d0; color: #166534;}
        .pill-warn {background: #fffbeb; border-color: #fde68a; color: #92400e;}
        .step-box {
            border-left: 4px solid #6366f1;
            background: #f8fafc;
            border-radius: 14px;
            padding: 0.82rem 1rem;
            margin-bottom: 0.55rem;
        }
        .step-box b {color: #0f172a;}
        .step-box span {color: #475569;}
        .section-title {font-size: 1.05rem; font-weight: 800; color: #0f172a; margin: 0.5rem 0 0.6rem 0;}
        .empty-state {
            border: 1px dashed #cbd5e1;
            border-radius: 22px;
            padding: 32px;
            background: #f8fafc;
            text-align: center;
            color: #475569;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 18px;
            padding: 0.85rem 1rem;
            box-shadow: 0 8px 20px rgba(15,23,42,0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card_metric(label: str, value: str, help_text: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="soft-card">
          <div class="mini-label">{label}</div>
          <div class="mini-value">{value}</div>
          {f'<div style="color:#64748b;font-size:0.86rem;margin-top:0.35rem;">{help_text}</div>' if help_text else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_path(final_state: dict) -> None:
    agent = (final_state.get("agent_used") or "unknown").lower()
    intent = final_state.get("intent") or "unknown"
    router = summarize_router_debug(final_state)
    mem = memory_summary(final_state)
    tools = final_state.get("tool_results") or []
    path = [
        ("User", "Query received", "pill-muted"),
        ("Supervisor", f"Intent: {intent}", "pill"),
        ("Memory", f"{mem['recent_message_count']} msg / {mem['entity_count']} facts", "pill-ok" if mem["context_text_available"] else "pill-muted"),
        (AGENT_DESCRIPTIONS.get(agent, {}).get("label", agent.title()), "Selected agent", "pill"),
        ("Tool", f"{len(tools)} result(s)", "pill-ok" if tools else "pill-muted"),
        ("Response", "Ready", "pill-ok" if final_state.get("response") else "pill-warn"),
    ]
    pills = "".join(f'<span class="pill {klass}">{name} · {detail}</span>' for name, detail, klass in path)
    st.markdown(pills, unsafe_allow_html=True)
    st.caption(
        f"Router: {router.get('router_method')}"
        + (f" · confidence {router.get('confidence')}" if router.get("confidence") is not None else "")
        + (" · fallback used" if router.get("fallback_used") else "")
    )


def render_answer(final_state: dict) -> None:
    response = final_state.get("response") or "No response returned."
    st.markdown(
        f"""
        <div class="answer-card">
            <h3>Final answer</h3>
            <div>{response}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_timeline(final_state: dict) -> None:
    for row in build_timeline_rows(final_state):
        st.markdown(
            f"""
            <div class="step-box">
              <b>{row['step']} · {row['stage']}</b><br/>
              <span>{row['detail']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_evidence(final_state: dict) -> None:
    agent = final_state.get("agent_used")
    tools = final_state.get("tool_results") or []

    if agent == "knowledge":
        st.markdown("### RAG Workflow Inspector")
        st.markdown("#### 1. Query processing")
        st.dataframe(rag_query_processing_rows(final_state), use_container_width=True, hide_index=True)
        st.markdown("#### 2. Access control")
        st.dataframe(rag_access_rows(final_state), use_container_width=True, hide_index=True)
        st.markdown("#### 3. Ranking explanation")
        mapping = [("BM25 results", "bm25_results"), ("Vector results", "vector_results"), ("RRF fused results", "rrf_fused_results"), ("Final selected chunks", "final_selected_chunks")]
        for title, key in mapping:
            with st.expander(title, expanded=False):
                rows = rag_ranking_debug_rows(final_state, key)
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No {title.lower()} available.")
        retrieval_rows = rag_retrieval_results_rows(final_state)
        if retrieval_rows:
            st.dataframe(retrieval_rows, use_container_width=True, hide_index=True)
        st.markdown("#### 4. Generated answer grounding")
        st.dataframe(rag_generation_rows(final_state), use_container_width=True, hide_index=True)

        citations = final_state.get("citations") or []
        used_chunks = final_state.get("used_chunks") or []
        if citations:
            st.markdown('<div class="section-title">Citations</div>', unsafe_allow_html=True)
            st.dataframe(citations_to_rows(citations), use_container_width=True, hide_index=True)
        if used_chunks:
            st.markdown('<div class="section-title">Used chunks</div>', unsafe_allow_html=True)
            st.dataframe(used_chunks_to_rows(used_chunks), use_container_width=True, hide_index=True)
        if not citations and not used_chunks:
            st.info("No citations or chunks were returned for this knowledge response.")

    elif agent == "research":
        rows = research_results_to_rows(tools)
        if rows:
            st.markdown('<div class="section-title">Research results</div>', unsafe_allow_html=True)
            st.dataframe(rows, use_container_width=True, hide_index=True)
        elif tools:
            st.info("Research tool ran, but no result rows were available.")
        else:
            st.info("No research tool results were returned.")

    elif agent == "action":
        if tools:
            st.markdown('<div class="section-title">Workflow tool results</div>', unsafe_allow_html=True)
            st.dataframe(tool_results_to_rows(tools), use_container_width=True, hide_index=True)
        else:
            st.info("No action tool was executed for this request.")
    else:
        st.info("No evidence or tool output is available for this route.")

    if tools:
        with st.expander("Raw tool payload", expanded=False):
            st.json(safe_jsonable(tools))


def render_memory(final_state: dict) -> None:
    mem = memory_summary(final_state)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loaded messages", mem["recent_message_count"])
    c2.metric("Known facts", mem["entity_count"])
    c3.metric("After response", mem["memory_message_count"])
    c4.metric("Context", "Available" if mem["context_text_available"] else "Empty")

    context_text = final_state.get("memory_context_text") or ""
    if context_text:
        st.markdown('<div class="section-title">Memory context text</div>', unsafe_allow_html=True)
        st.code(context_text, language="text")
    else:
        st.info("No memory context text was available for this run.")

    left, right = st.columns(2)
    with left:
        with st.expander("Recent messages loaded", expanded=False):
            st.json(safe_jsonable((final_state.get("memory_context") or {}).get("recent_messages", [])))
    with right:
        with st.expander("Known entities loaded", expanded=False):
            st.json(safe_jsonable((final_state.get("memory_context") or {}).get("entities", [])))


inject_styles()

st.markdown(
    """
    <div class="cortex-hero">
        <h1>🧠 Cortex Agent Command Center</h1>
        <p>A clean operator view for a multi-agent RAG system: route the query, inspect the answer, verify evidence, check memory, and run retrieval evaluation without drowning in raw debug output.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Ask Cortex")
    query = st.text_area(
        "User query",
        value="",
        height=132,
        placeholder="Example: What is the parental leave policy?",
    )
    user_tier = st.radio("Access tier", ["standard", "manager", "exec"], horizontal=True)
    user_id = st.text_input("User ID", value="demo-user")

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"session_{uuid4().hex[:12]}"
    session_id = st.text_input("Session ID", value=st.session_state["session_id"])
    st.session_state["session_id"] = session_id or st.session_state["session_id"]

    example_query = st.selectbox(
        "Try a scenario",
        [
            "",
            "What is the parental leave policy?",
            "Create a support ticket for my VPN issue.",
            "Escalate this issue, it is blocking production.",
            "Find recent AI governance trends.",
            "What is the executive strategy for APAC expansion?",
        ],
        index=0,
    )
    if example_query and not query:
        query = example_query

    run_agent = st.button("Run Cortex", type="primary", use_container_width=True)

    with st.expander("Advanced", expanded=False):
        top_k = st.slider("Evaluation top_k", min_value=1, max_value=20, value=5)
        data_dir = st.text_input("Data directory", value="data")
        registry_db_path = st.text_input("Registry DB", value="data/cortex_registry.sqlite")
        use_registry = True

    st.caption("Diagnostics are intentionally tucked into tabs/expanders to keep the main run readable.")

if "final_state" not in st.session_state:
    st.session_state["final_state"] = None

if run_agent:
    if not query.strip():
        st.warning("Enter a query before running Cortex.")
    else:
        try:
            with st.spinner("Routing through Cortex..."):
                final_state = invoke_with_trace(
                    query=query,
                    user_id=user_id,
                    user_tier=user_tier,
                    session_id=session_id,
                )
            st.session_state["final_state"] = final_state
        except Exception as exc:
            st.error(f"Agent invocation failed: {exc}")

run_tab, evidence_tab, memory_tab, eval_tab, debug_tab = st.tabs(
    ["Run", "Evidence", "Memory", "Evaluation", "Developer Console"]
)

with run_tab:
    final_state = st.session_state.get("final_state")
    if not final_state:
        st.markdown(
            """
            <div class="empty-state">
                <h3>Start with one query</h3>
                <p>Run Cortex from the sidebar. The main screen will show only the answer and the essential route. Evidence, memory, and raw debug stay one click away.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-title">System map</div>', unsafe_allow_html=True)
        st.dataframe(build_agent_capability_rows(), use_container_width=True, hide_index=True)
    else:
        summary = build_run_summary(final_state)
        agent = summary["agent"]
        agent_meta = AGENT_DESCRIPTIONS.get(agent, {"icon": "🤖", "label": agent.title(), "summary": "Agent execution complete."})

        c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.0, 1.0, 1.4])
        with c1:
            card_metric("Agent", f"{agent_meta['icon']} {agent_meta['label']}", agent_meta["summary"])
        with c2:
            card_metric("Intent", summary["intent"], build_routing_explanation(summary["intent"], agent))
        with c3:
            card_metric("Cost", summary["cost"])
        with c4:
            card_metric("Tools", str(summary["tools"]))
        with c5:
            card_metric("Memory", summary["memory"], "recent messages / known facts")

        render_answer(final_state)
        render_agent_path(final_state)

        left, right = st.columns([1.1, 0.9])
        with left:
            st.markdown('<div class="section-title">Execution story</div>', unsafe_allow_html=True)
            render_compact_timeline(final_state)
        with right:
            st.markdown('<div class="section-title">Agent details</div>', unsafe_allow_html=True)
            detail_rows = build_agent_detail_rows(final_state)
            for row in detail_rows:
                st.markdown(f"**{row['label']}**")
                st.caption(truncate_text(row.get("value"), 260))

with evidence_tab:
    final_state = st.session_state.get("final_state")
    if not final_state:
        st.info("Run Cortex first to inspect evidence and tool outputs.")
    else:
        render_evidence(final_state)

with memory_tab:
    final_state = st.session_state.get("final_state")
    if not final_state:
        st.info("Run Cortex first to inspect memory usage.")
    else:
        render_memory(final_state)

with eval_tab:
    st.markdown('<div class="section-title">Golden dataset evaluation</div>', unsafe_allow_html=True)
    left, right = st.columns([0.72, 0.28])
    with left:
        dataset_path = st.text_input("Golden dataset", value="evaluation/golden_dataset.json")
        eval_data_dir = st.text_input("Evaluation data directory", value=data_dir, key="eval_data_dir")
    with right:
        eval_top_k = st.slider("top_k", min_value=1, max_value=20, value=top_k, key="eval_top_k")
        st.caption("Registry: always enabled")
        st.caption("PGVector: always enabled")
    eval_mode = st.selectbox("Evaluation mode", ["local_hash + word_window", "local_hash + section", "openai + pgvector + section"])
    eval_registry_db_path = st.text_input("Evaluation registry DB", value=registry_db_path, key="eval_registry_db_path")
    eval_database_url = st.text_input("Evaluation DATABASE_URL override (optional)", value="")
    st.caption(f"DATABASE_URL configured: {'yes' if bool(eval_database_url or os.getenv('DATABASE_URL', '')) else 'no'}")
    run_eval = st.button("Run retrieval evaluation", type="primary")

    if run_eval:
        dataset_file = Path(dataset_path)
        eval_data_path = Path(eval_data_dir)
        if not eval_data_path.exists():
            st.error(f"Data directory '{eval_data_dir}' does not exist.")
        elif not dataset_file.exists():
            st.error(f"Golden dataset '{dataset_path}' does not exist.")
        else:
            eval_registry_path = Path(eval_registry_db_path)
            eval_registry_for_call = str(eval_registry_path) if eval_registry_path.exists() else None
            try:
                with st.spinner("Running retrieval evaluation..."):
                    summary = evaluate_retrieval(
                        dataset_path=dataset_path,
                        data_dir=eval_data_dir,
                        registry_db_path=eval_registry_for_call,
                        top_k=eval_top_k,
                        use_pgvector=True,
                        database_url=eval_database_url or None,
                    )
                k = summary.get("top_k", eval_top_k)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Cases", summary.get("case_count", 0))
                m2.metric(f"P@{k}", round(summary.get("precision_at_k", 0.0), 4))
                m3.metric("MRR", round(summary.get("mrr", 0.0), 4))
                m4.metric(f"NDCG@{k}", round(summary.get("ndcg_at_k", 0.0), 4))

                cases = summary.get("cases", [])
                if cases:
                    st.markdown('<div class="section-title">Per-case results</div>', unsafe_allow_html=True)
                    st.dataframe(cases, use_container_width=True, hide_index=True)
                with st.expander("Raw evaluation summary", expanded=False):
                    st.json(safe_jsonable(summary))
                with st.expander("Retrieval config", expanded=False):
                    st.json(safe_jsonable(summary.get("retrieval_config", {})))
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

with debug_tab:
    final_state = st.session_state.get("final_state")
    if not final_state:
        st.info("Run Cortex first to inspect developer diagnostics.")
    else:
        router_debug = get_router_debug(final_state)
        agent_debug = get_agent_debug(final_state)
        st.markdown('<div class="section-title">Router debug</div>', unsafe_allow_html=True)
        st.json(safe_jsonable(router_debug))
        st.markdown('<div class="section-title">Agent debug</div>', unsafe_allow_html=True)
        st.json(safe_jsonable(agent_debug))
        with st.expander("Full final state", expanded=False):
            st.json(safe_jsonable(final_state))
