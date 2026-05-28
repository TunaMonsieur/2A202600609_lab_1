"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import sys
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1K OUTPUT tokens (USD) — update if pricing changes
# ---------------------------------------------------------------------------
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-4o": 0.010,
    "gpt-4o-mini": 0.0006,
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Task 1 — Call GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API and return the response text + latency.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start_time
    response_text = response.choices[0].message.content or ""
    return response_text, latency


# ---------------------------------------------------------------------------
# Task 2 — Call GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API using gpt-4o-mini and return the
    response text + latency.

    Args:
        prompt:      The user message to send.
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        Reuse call_openai() by passing model=OPENAI_MINI_MODEL.
    """
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 3 — Compare GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call both gpt-4o and gpt-4o-mini with the same prompt and return a
    comparison dictionary.

    Args:
        prompt: The user message to send to both models.

    Returns:
        A dict with keys:
            - "gpt4o_response":      str
            - "mini_response":       str
            - "gpt4o_latency":       float
            - "mini_latency":        float
            - "gpt4o_cost_estimate": float  (estimated USD for the response)

    Hint:
        Cost estimate = (len(response.split()) / 0.75) / 1000 * COST_PER_1K_OUTPUT_TOKENS["gpt-4o"]
        (0.75 words ≈ 1 token is a rough approximation)
    """
    gpt4o_response, gpt4o_latency = call_openai(prompt)
    mini_response, mini_latency = call_openai_mini(prompt)

    estimated_output_tokens = (len(gpt4o_response.split()) / 0.75)
    gpt4o_cost_estimate = (
        estimated_output_tokens / 1000
    ) * COST_PER_1K_OUTPUT_TOKENS["gpt-4o"]

    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate,
    }


# ---------------------------------------------------------------------------
# Task 4 — Streaming chatbot with conversation history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal.

    Behaviour:
        - Streams tokens from OpenAI as they arrive (print each chunk).
        - Maintains the last 3 conversation turns in history.
        - Typing 'quit' or 'exit' ends the loop.

    Hints:
        - Keep a list `history` of {"role": ..., "content": ...} dicts.
        - Use stream=True in client.chat.completions.create() and iterate:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
        - After each turn, append the assistant reply to history.
        - Trim history to the last 3 turns: history = history[-3:]
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history: list[dict[str, str]] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})
        messages = history[-3:]

        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True,
        )

        assistant_text = ""
        print("Assistant: ", end="", flush=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            assistant_text += delta
            print(delta, end="", flush=True)
        print()

        history.append({"role": "assistant", "content": assistant_text})
        history = history[-3:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception:
            if attempt >= max_retries:
                raise
            delay = base_delay * (2**attempt)
            time.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []
    for prompt in prompts:
        comparison = compare_models(prompt)
        comparison_with_prompt = {"prompt": prompt, **comparison}
        results.append(comparison_with_prompt)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of compare_models results as a readable text table.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A formatted string table with columns:
        Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency

    Hint:
        Truncate long text to 40 characters for readability.
    """
    headers = [
        "Prompt",
        "GPT-4o Response",
        "Mini Response",
        "GPT-4o Latency",
        "Mini Latency",
    ]

    def truncate(text: str, max_len: int = 40) -> str:
        if len(text) <= max_len:
            return text
        return f"{text[: max_len - 3]}..."

    rows = [headers]
    for result in results:
        rows.append(
            [
                truncate(str(result.get("prompt", ""))),
                truncate(str(result.get("gpt4o_response", ""))),
                truncate(str(result.get("mini_response", ""))),
                f"{float(result.get('gpt4o_latency', 0.0)):.3f}s",
                f"{float(result.get('mini_latency', 0.0)):.3f}s",
            ]
        )

    col_widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]

    def format_row(row: list[str]) -> str:
        return " | ".join(
            row[i].ljust(col_widths[i]) for i in range(len(col_widths))
        )

    separator = "-+-".join("-" * width for width in col_widths)
    table_lines = [format_row(rows[0]), separator]
    table_lines.extend(format_row(row) for row in rows[1:])
    return "\n".join(table_lines)


_SAFE_MODULE_ALIAS = "day01_lab_assignment_template"
sys.modules.setdefault(_SAFE_MODULE_ALIAS, sys.modules[__name__])
compare_models.__module__ = _SAFE_MODULE_ALIAS


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
def _print_banner() -> None:
    title = "LLM Playground - Day 1 Lab"
    width = 64
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)
    print("1) Compare GPT-4o and GPT-4o-mini")
    print("2) Start streaming chatbot")
    print("3) Exit")
    print("-" * width)


def _run_compare_ui() -> None:
    prompt = input("\nEnter prompt to compare: ").strip()
    if not prompt:
        print("Prompt cannot be empty.")
        return

    print("\nRunning comparison...")
    result = compare_models(prompt)
    print("\nResult")
    print("-" * 64)
    print(f"GPT-4o latency     : {result['gpt4o_latency']:.3f}s")
    print(f"GPT-4o-mini latency: {result['mini_latency']:.3f}s")
    print(f"GPT-4o est. cost   : ${result['gpt4o_cost_estimate']:.6f}")
    print("-" * 64)
    print(f"GPT-4o response:\n{result['gpt4o_response']}\n")
    print(f"GPT-4o-mini response:\n{result['mini_response']}")
    print("-" * 64)


if __name__ == "__main__":
    while True:
        _print_banner()
        choice = input("Choose an option (1-3): ").strip()

        if choice == "1":
            _run_compare_ui()
        elif choice == "2":
            print("\nChatbot mode. Type 'quit' or 'exit' to return.\n")
            streaming_chatbot()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please choose 1, 2, or 3.")
