"""Streamlit interface:  streamlit run app.py   (needs MISTRAL_API_KEY in .env)"""
import json

import streamlit as st

from agent.agent import build_default_agent

st.set_page_config(page_title="Industrial Sensor Query Agent", page_icon="⚙️")
st.title("⚙️ Industrial Sensor Query Agent")
st.caption("Ask about the AI4I 2020 predictive-maintenance data. Numbers come from exact pandas "
           "calculations, explanations from the documentation, and the steps are shown below each answer.")

EXAMPLES = [
    "How many machines failed?",
    "What is the average torque of failed machines compared with normal ones?",
    "Which product type has the highest failure rate?",
    "How many machines with tool wear above 200 minutes failed?",
    "What causes a heat dissipation failure?",
]


@st.cache_resource
def get_agent():
    return build_default_agent()


try:
    with st.spinner("Loading the agent (first start embeds the documentation)…"):
        agent = get_agent()
except Exception as exc:
    st.error(f"Could not start the agent: {exc}. Is MISTRAL_API_KEY set in your .env file?")
    st.stop()

with st.sidebar:
    st.subheader("Example questions")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state["question"] = q

question = st.text_input("Your question", key="question",
                         placeholder="e.g. What is the median rotational speed of failed machines?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Thinking…"):
        result = agent.ask(question)
    st.subheader("Answer")
    st.write(result.answer)
    with st.expander(f"How this was answered ({len(result.steps)} tool call(s))"):
        if not result.steps:
            st.write("The model answered without calling a tool.")
        for i, s in enumerate(result.steps, 1):
            st.markdown(f"**{i}. `{s.tool}`** with `{json.dumps(s.args)}`")
            st.json(s.result)
