"""
Streamlit UI for Day 1 LLM lab.

This module also re-exports required functions for the test suite.
"""

from __future__ import annotations

import os
from typing import Generator

from solution.solution import (
    OPENAI_MODEL,
    call_openai,
    call_openai_mini,
    compare_models,
    format_comparison_table,
    retry_with_backoff,
    streaming_chatbot,
)


def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Local wrapper so tests can patch app.compare_models.
    """
    results = []
    for prompt in prompts:
        comparison = compare_models(prompt)
        results.append({"prompt": prompt, **comparison})
    return results


def stream_openai_reply(prompt: str, history: list[dict[str, str]]) -> Generator[str, None, str]:
    """
    Stream assistant response chunks and return the final text.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = history + [{"role": "user", "content": prompt}]
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages[-6:],
        stream=True,
    )

    collected: list[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            collected.append(delta)
            yield delta


def _run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="LLM Lab UI", page_icon="🤖", layout="wide")
    st.title("🤖 LLM Playground - Day 1")
    st.caption("So sánh GPT-4o và GPT-4o-mini + chatbot streaming")

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Chưa có OPENAI_API_KEY trong môi trường.")

    tab_compare, tab_chat = st.tabs(["So sánh model", "Chatbot streaming"])

    with tab_compare:
        prompt = st.text_area(
            "Nhập prompt",
            value="Explain the difference between temperature and top_p in one sentence.",
            height=120,
        )
        if st.button("So sánh", type="primary", use_container_width=True):
            if not prompt.strip():
                st.error("Prompt không được để trống.")
            else:
                with st.spinner("Đang gọi model..."):
                    result = compare_models(prompt.strip())

                c1, c2, c3 = st.columns(3)
                c1.metric("GPT-4o latency", f"{result['gpt4o_latency']:.3f}s")
                c2.metric("Mini latency", f"{result['mini_latency']:.3f}s")
                c3.metric("GPT-4o est. cost", f"${result['gpt4o_cost_estimate']:.6f}")

                left, right = st.columns(2)
                left.subheader("GPT-4o")
                left.write(result["gpt4o_response"])
                right.subheader("GPT-4o-mini")
                right.write(result["mini_response"])

    with tab_chat:
        st.write("Chat thời gian thực với GPT-4o (giữ lịch sử gần).")
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_prompt = st.chat_input("Nhập tin nhắn...")
        if user_prompt:
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                history = st.session_state.chat_messages[-6:-1]
                placeholder = st.empty()
                full_text = ""
                for piece in stream_openai_reply(user_prompt, history):
                    full_text += piece
                    placeholder.markdown(full_text)

            st.session_state.chat_messages.append({"role": "assistant", "content": full_text})

        if st.button("Xóa lịch sử chat"):
            st.session_state.chat_messages = []
            st.rerun()


if __name__ == "__main__":
    _run_streamlit()
