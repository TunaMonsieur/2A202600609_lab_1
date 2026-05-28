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

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(79, 70, 229, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 24%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .hero {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 24px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.15rem;
            line-height: 1.1;
        }
        .hero p {
            margin: 0.5rem 0 0;
            color: #475569;
            font-size: 1.02rem;
        }
        .feature-card {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            height: 100%;
        }
        .feature-card h3 {
            margin: 0 0 0.4rem;
            font-size: 1rem;
        }
        .feature-card p {
            margin: 0;
            color: #475569;
            font-size: 0.94rem;
        }
        .stButton > button {
            border-radius: 999px;
            padding: 0.65rem 1.1rem;
            font-weight: 600;
        }
        /* Ensure readable text color on all backgrounds (fix white-on-white issues) */
        .stApp, .main .block-container, .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp div, .stApp span {
            color: #0f172a !important;
        }
        .stApp a, a {
            color: #0369a1 !important;
        }
        /* Sidebar and uploader readability fixes */
        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.98) !important;
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] * {
            color: #0f172a !important;
        }
        [data-testid="stFileUploader"], .stFileUploader, [data-testid="stFileUpload"] {
            background: rgba(255,255,255,0.98) !important;
            color: #0f172a !important;
        }
        [data-testid="stFileUploader"] small, .stFileUploader small {
            color: #0f172a !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>🤖 LLM Playground - Day 1</h1>
            <p>So sánh GPT-4o và GPT-4o-mini, rồi chat streaming theo kiểu trực tiếp và dễ theo dõi hơn.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sidebar = st.sidebar
    sidebar.title("Hướng dẫn nhanh")
    sidebar.markdown(
        """
        1. Nhập prompt để so sánh 2 model.
        2. Mở tab chat để thử phản hồi streaming.
        3. Nếu chưa thấy kết quả, kiểm tra "OPENAI_API_KEY".
        """
    )
    sidebar.divider()
    sidebar.caption("Mẹo: dùng prompt ngắn, rõ mục tiêu để so sánh dễ hơn.")

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Chưa có OPENAI_API_KEY trong môi trường. Ứng dụng vẫn mở được giao diện nhưng sẽ không gọi được API.")

    info_left, info_mid, info_right = st.columns(3)
    info_left.markdown(
        '<div class="feature-card"><h3>So sánh model</h3><p>Nhìn latency, ước tính chi phí và nội dung trả lời đặt cạnh nhau.</p></div>',
        unsafe_allow_html=True,
    )
    info_mid.markdown(
        '<div class="feature-card"><h3>Chat streaming</h3><p>Quan sát câu trả lời xuất hiện dần, giống trải nghiệm thật hơn.</p></div>',
        unsafe_allow_html=True,
    )
    info_right.markdown(
        '<div class="feature-card"><h3>Giao diện gọn</h3><p>Bố cục rõ ràng, ưu tiên thao tác nhanh và ít rối mắt.</p></div>',
        unsafe_allow_html=True,
    )

    # Sidebar task selector: switch between compare, chat, and batch compare
    task = sidebar.radio("Chọn tác vụ", ["So sánh model", "Chatbot streaming", "Batch compare"], index=0)

    if task == "So sánh model":
        left, right = st.columns([1.15, 0.85], gap="large")

        with left:
            st.subheader("Nhập prompt")
            st.caption("Ví dụ: hỏi cùng một yêu cầu để nhìn sự khác biệt giữa tốc độ và chất lượng trả lời.")
            with st.form("compare_form", clear_on_submit=False):
                prompt = st.text_area(
                    "Prompt",
                    value="Explain the difference between temperature and top_p in one sentence.",
                    height=150,
                    placeholder="Nhập yêu cầu muốn so sánh...",
                )
                submitted = st.form_submit_button("So sánh ngay", type="primary", use_container_width=True)

        with right:
            st.subheader("Kết quả sẽ hiển thị ở đây")
            st.info("Sau khi bấm so sánh, bạn sẽ thấy latency, chi phí ước tính và câu trả lời của từng model.")

        if submitted:
            if not prompt.strip():
                st.error("Prompt không được để trống.")
            else:
                with st.spinner("Đang gọi model..."):
                    result = compare_models(prompt.strip())

                c1, c2, c3 = st.columns(3)
                c1.metric("GPT-4o latency", f"{result['gpt4o_latency']:.3f}s")
                c2.metric("Mini latency", f"{result['mini_latency']:.3f}s")
                c3.metric("GPT-4o est. cost", f"${result['gpt4o_cost_estimate']:.6f}")

                left_result, right_result = st.columns(2, gap="large")
                with left_result:
                    st.markdown(
                        '<div class="feature-card"><h3>GPT-4o</h3><p style="white-space: pre-wrap;">' +
                        str(result["gpt4o_response"]) +
                        '</p></div>',
                        unsafe_allow_html=True,
                    )
                with right_result:
                    st.markdown(
                        '<div class="feature-card"><h3>GPT-4o-mini</h3><p style="white-space: pre-wrap;">' +
                        str(result["mini_response"]) +
                        '</p></div>',
                        unsafe_allow_html=True,
                    )

    elif task == "Chatbot streaming":
        chat_left, chat_right = st.columns([0.72, 0.28], gap="large")

        with chat_left:
            st.subheader("Chat thời gian thực")
            st.caption("Trò chuyện với model và xem câu trả lời xuất hiện theo từng đoạn.")

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

        with chat_right:
            st.subheader("Quản lý chat")
            st.write("Dùng nút dưới đây để xoá toàn bộ hội thoại và bắt đầu lại từ đầu.")
            if st.button("Xóa lịch sử chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()
            st.divider()
            st.caption("Gợi ý prompt")
            st.write("• Giải thích một khái niệm trong 2 câu.")
            st.write("• Tóm tắt nội dung bằng tiếng Việt.")
            st.write("• Đưa ra 3 ý chính và ví dụ ngắn.")

    else:  # Batch compare
        st.subheader("Batch compare")
        st.caption("Tải lên file .txt (mỗi dòng 1 prompt) hoặc dán nhiều prompt xuống dưới.")
        with st.form("batch_form"):
            uploaded = st.file_uploader("Upload prompts (.txt)", type=["txt"])
            pasted = st.text_area("Hoặc dán prompts (mỗi prompt 1 dòng)", height=200)
            run_batch = st.form_submit_button("Chạy batch", type="primary", use_container_width=True)

        if run_batch:
            prompts: list[str] = []
            if uploaded is not None:
                try:
                    raw = uploaded.read().decode("utf-8")
                except Exception:
                    raw = uploaded.read().decode("latin-1")
                prompts = [line.strip() for line in raw.splitlines() if line.strip()]
            else:
                prompts = [line.strip() for line in pasted.splitlines() if line.strip()]

            if not prompts:
                st.error("Không có prompt nào để chạy.")
            else:
                with st.spinner(f"Chạy compare cho {len(prompts)} prompt... này có thể mất một lúc"):
                    results = batch_compare(prompts)

                st.success(f"Hoàn thành: {len(results)} prompt")
                st.subheader("Bảng tóm tắt")
                st.code(format_comparison_table(results))


if __name__ == "__main__":
    _run_streamlit()
