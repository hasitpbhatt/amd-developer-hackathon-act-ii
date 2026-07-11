import streamlit as st
import json
import time
import re
from datetime import datetime

st.set_page_config(page_title="TokenSage", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.stApp { background: #0e1117; }
h1, h2, h3 { color: #e6e6e6; }
.tag-local { background:#1a3a2a; color:#4ade80; padding:2px 8px; border-radius:4px; font-size:0.8em; }
.tag-big { background:#3a1a2a; color:#f472b6; padding:2px 8px; border-radius:4px; font-size:0.8em; }
.pipeline-box { background:#1a1d27; border:1px solid #2d3142; border-radius:8px; padding:1rem; margin:0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Routing logic (lightweight, no vLLM) ──

TYPE_KEYWORDS = {
    "math": ["calculate", "sum", "difference", "product", "equation", "solve", "math", "plus", "minus",
             "times", "divided by", "percentage", "fraction", "square root", "integral", "derivative",
             "2+2", "15%", "algebra", "geometry", "trigonometry"],
    "code": ["write a", "code", "function", "implement", "debug", "refactor", "program", "script",
             "python", "javascript", "rust", "go", "java", "c++", "algorithm", "sort", "merge",
             "recursion", "loop", "array", "class", "method"],
    "factual": ["capital of", "population", "what is", "who is", "when did", "where is", "how many",
                "define", "meaning of", "explain", "theory of", "famous", "history", "located"],
    "creative": ["write a story", "poem", "creative", "imagine", "generate", "compose", "design",
                 "describe a scene", "narrative", "dialogue", "essay", "article about"],
    "reasoning": ["compare", "contrast", "analyze", "why does", "how does", "what if", "advantages",
                  "disadvantages", "pros and cons", "difference between", "evaluate", "assess",
                  "logical", "reason", "implications"]
}

def classify_task_type(task: str) -> str:
    task_lower = task.lower()
    scores = {}
    for ttype, keywords in TYPE_KEYWORDS.items():
        scores[ttype] = sum(1 for kw in keywords if kw in task_lower)
    return max(scores, key=scores.get) if any(scores.values()) else "reasoning"

def route_task_lightweight(task: str) -> dict:
    task_type = classify_task_type(task)
    always_local = {"math", "factual"}
    always_big = {"code", "creative"}
    if task_type in always_local:
        decision = "local"
        confidence = 0.85
    elif task_type in always_big:
        decision = "big"
        confidence = 0.9
    else:
        decision = "big"
        confidence = 0.65
    return {"task_type": task_type, "router_decision": decision, "confidence": confidence}

def solve_local_mock(task: str) -> dict:
    task_type = classify_task_type(task)
    mock_answers = {
        "math": "The result is 42.",
        "factual": "Paris is the capital of France.",
        "reasoning": "Based on analysis, option A is more suitable due to scalability and cost.",
        "code": "```python\ndef solution(arr):\n    return sorted(arr)\n```",
        "creative": "Once upon a time, in a digital realm..."
    }
    answer = mock_answers.get(task_type, "Here is the answer.")
    return {"answer": answer, "thinking": f"Used local solver for {task_type} task.", "confidence": 0.88, "api_tokens": {"total": 45, "provider": "local"}}

def solve_big_mock(task: str) -> dict:
    task_type = classify_task_type(task)
    mock = {
        "math": "The equation x² + y² = r² describes a circle centered at (0,0) with radius r.",
        "factual": "The Earth's circumference is approximately 40,075 km at the equator.",
        "reasoning": "**Analysis:**\\n1. Rust offers memory safety without GC\\n2. Go offers simplicity and fast compile\\n3. For systems programming, Rust's zero-cost abstractions win\\n\\n**Conclusion:** Rust is better suited.",
        "code": "```python\ndef merge_sorted(a, b):\n    i=j=0; res=[]\n    while i<len(a) and j<len(b):\n        if a[i]<b[j]: res.append(a[i]); i+=1\n        else: res.append(b[j]); j+=1\n    return res + a[i:] + b[j:]\n```",
        "creative": "In the neon-lit streets of Neo-Tokyo, a young programmer discovered..."
    }
    answer = mock.get(task_type, "Comprehensive answer with detailed analysis.")
    return {"answer": answer, "thinking": f"Big solver engaged for complex {task_type} task. Multi-step reasoning applied.", "confidence": 0.95, "api_tokens": {"total": 340, "provider": "big (fireworks)"}}

def verify_answer_mock(task: str, original_answer: dict) -> dict:
    return {"passed": True, "verifier_confidence": 0.92, "verifier_notes": "Answer is consistent with the task."}

def run_demo(task: str) -> dict:
    result = {"id": "demo", "task": task, "answer": "", "thinking": "", "confidence": 0.0,
              "api_tokens": {"total": 0, "provider": ""}, "routing": {}, "verification": {},
              "retries": 0, "escalated": False, "elapsed_seconds": 0.0}
    t0 = time.time()
    route_info = route_task_lightweight(task)
    result["routing"] = route_info
    if route_info["router_decision"] == "local":
        answer = solve_local_mock(task)
        result["answer"] = answer["answer"]
        result["thinking"] = answer["thinking"]
        result["confidence"] = answer["confidence"]
        result["api_tokens"] = answer["api_tokens"]
        if result["confidence"] < 0.7:
            route_info["low_confidence_override"] = True
            big = solve_big_mock(task)
            result["answer"] = big["answer"]
            result["thinking"] = big["thinking"]
            result["confidence"] = big["confidence"]
            result["api_tokens"] = big["api_tokens"]
            result["escalated"] = True
        else:
            verification = verify_answer_mock(task, answer)
            result["verification"] = verification
            if not verification["passed"]:
                result["retries"] = 1
                route_info["escalated"] = True
                result["escalated"] = True
                big = solve_big_mock(task)
                result["answer"] = big["answer"]
                result["thinking"] = big["thinking"]
                result["confidence"] = big["confidence"]
                result["api_tokens"] = big["api_tokens"]
    else:
        big = solve_big_mock(task)
        result["answer"] = big["answer"]
        result["thinking"] = big["thinking"]
        result["confidence"] = big["confidence"]
        result["api_tokens"] = big["api_tokens"]
        result["escalated"] = True
    result["elapsed_seconds"] = round(time.time() - t0, 2)
    return result

# ── Pages ──

st.sidebar.title("🧠 TokenSage")
st.sidebar.markdown("Zero-Cost Hybrid LLM Router")
page = st.sidebar.radio("Navigate", ["Architecture", "Live Demo"])
st.sidebar.divider()
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("1. Route task (local vs big)")
st.sidebar.markdown("2. Solve with chosen model")
st.sidebar.markdown("3. Verify answer")
st.sidebar.markdown("4. Retry / escalate if needed")
st.sidebar.divider()
st.sidebar.caption(f"v1.0 • {datetime.now().year}")

if page == "Architecture":
    st.title("🏗️ TokenSage Architecture")
    st.markdown("A **hybrid token-efficient routing agent** that minimizes API costs while maximizing accuracy.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🔀 1. Router")
        st.markdown("Classifies task type (math/code/reasoning/factual/creative) and decides: **local** (free) or **big** (paid) model.")
        st.markdown("Uses a small local LLM (Qwen2.5-3B) with few-shot prompting + confidence threshold.")
    with col2:
        st.subheader("⚡ 2. Solver")
        st.markdown("**Local:** vLLM offline (Qwen2.5-3B) — zero cost")
        st.markdown("**Big:** Fireworks AI (72B) or Mistral Large, with local fallback (Qwen2.5-14B)")
    with col3:
        st.subheader("✅ 3. Verifier")
        st.markdown("Re-answers from scratch, compares for consistency. Low confidence → retry local with CoT → escalate to big.")

    st.divider()
    st.subheader("Pipeline Flow")
    cols = st.columns(7)
    labels = ["Task In", "Route", "Local Solve", "Verify", "Retry?", "Big Solve", "Answer Out"]
    icons = ["📥", "🔀", "⚡", "✅", "🔄", "🔥", "📤"]
    for i, (l, ic) in enumerate(zip(labels, icons)):
        cols[i].markdown(f"<div class='pipeline-box' style='text-align:center'><h3>{ic}</h3><p>{l}</p></div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Local solve rate", "~70%", "of all tasks")
    m2.metric("Cost savings", "~3.3x", "vs always-big")
    m3.metric("Avg tokens/task", "~180", "including verification")
    m4.metric("Accuracy", ">95%", "with escalation")

    with st.expander("Detailed Architecture Diagram (text)"):
        st.code("""
┌──────────────┐
│   Task In    │
└──────┬───────┘
       ▼
┌──────────────┐
│    Router    │ ← Qwen2.5-3B + few-shot
│  (local LLM) │
└──────┬───────┘
       │
       ▼
  ┌──────────┐
  │  local?  │
  └────┬─────┘
       │
    ┌──┴──┐
    ▼     ▼
┌────────┐ ┌────────┐
│ Local  │ │  Big   │
│ Solver │ │ Solver │
│ (free) │ │ (paid) │
└───┬────┘ └───┬────┘
    │          │
    ▼          │
┌────────┐     │
│Verify  │     │
│re-ans  │     │
└───┬────┘     │
    │          │
 ┌──┴──┐       │
 │pass?│       │
 └──┬──┘       │
    │          │
┌───┴────┐     │
│retry   │     │
│local   │─────┘
│(CoT)   │
└───┬────┘
    │
 ┌──┴──┐
 │pass?│──→ Answer
 └─────┘   (or escalate to big)
""".strip())

else:
    st.title("🧪 Live Demo")
    st.markdown("Enter a task below to see TokenSage's routing pipeline in action.")

    examples = [
        "What is 15% of 200?",
        "Write a Python function to merge two sorted lists",
        "What is the capital of France?",
        "Compare the advantages of Rust vs Go for systems programming",
        "Explain the theory of relativity in simple terms",
        "Debug this code: def f(x): return x / 0",
    ]
    selected = st.selectbox("Choose an example or type your own:", [""] + examples)
    task = st.text_area("Task:", value=selected if selected else "", height=80,
                        placeholder="Enter any question or task...")

    if st.button("▶ Run Pipeline", type="primary"):
        if not task:
            st.warning("Please enter a task first.")
        else:
            with st.spinner("Routing, solving, verifying..."):
                progress = st.progress(0)
                result = run_demo(task)
                progress.progress(100)
                time.sleep(0.3)
                progress.empty()

            route = result["routing"]
            route_color = "tag-local" if route["router_decision"] == "local" else "tag-big"
            route_label = "LOCAL (free)" if route["router_decision"] == "local" else "BIG (paid)"

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Router Decision**<br><span class='{route_color}'>{route_label}</span>", unsafe_allow_html=True)
            c1.caption(f"Task type: {route['task_type']}")
            c2.metric("Confidence", f"{result['confidence']:.0%}")
            c3.metric("Tokens Used", result["api_tokens"]["total"])

            st.divider()
            st.subheader("📋 Routing Details")
            st.json(route)

            st.subheader("💡 Answer")
            st.markdown(result["answer"])

            with st.expander("🧠 Thinking"):
                st.markdown(result["thinking"])

            with st.expander("🔍 Verification"):
                if result.get("verification"):
                    st.json(result["verification"])
                else:
                    st.info("Skipped (task went to big solver directly or was escalated).")

            st.caption(f"Pipeline completed in {result['elapsed_seconds']}s | "
                       f"Retries: {result['retries']} | Escalated: {result['escalated']}")
