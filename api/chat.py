"""
api/chat.py
-----------
LLM-powered chat interface for SentinelFraud.

The LLM (Gemini) receives a natural language message, decides whether to
call the /predict or /explain tool, and returns a plain-English response.

This is the "agentic AI" layer on top of the ML models.
"""

import os
import json
from typing import Optional

from google import genai
from google.genai import types

from .model_loader import predict_transaction, explain_transaction


# ---------- Gemini client (singleton) ----------

_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        _client = genai.Client(api_key=api_key)
    return _client


# ---------- Tool definitions ----------
# The LLM sees these as "functions it can call." When it decides to call one,
# Gemini returns the arguments, and we execute the Python function ourselves.

def predict_tool(
    Time: float,
    V1: float, V2: float, V3: float, V4: float, V5: float,
    V6: float, V7: float, V8: float, V9: float, V10: float,
    V11: float, V12: float, V13: float, V14: float, V15: float,
    V16: float, V17: float, V18: float, V19: float, V20: float,
    V21: float, V22: float, V23: float, V24: float, V25: float,
    V26: float, V27: float, V28: float, Amount: float,
) -> dict:
    """Score a credit-card transaction for fraud.

    Args:
        Time: Seconds since first transaction in dataset.
        V1-V28: PCA-transformed features (anonymized by the bank).
        Amount: Raw transaction amount.

    Returns:
        Dict with ensemble_score, is_fraud, threshold, and per-model scores.
    """
    tx = {
        "Time": Time,
        "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5,
        "V6": V6, "V7": V7, "V8": V8, "V9": V9, "V10": V10,
        "V11": V11, "V12": V12, "V13": V13, "V14": V14, "V15": V15,
        "V16": V16, "V17": V17, "V18": V18, "V19": V19, "V20": V20,
        "V21": V21, "V22": V22, "V23": V23, "V24": V24, "V25": V25,
        "V26": V26, "V27": V27, "V28": V28, "Amount": Amount,
    }
    return predict_transaction(tx)


def explain_tool(
    Time: float,
    V1: float, V2: float, V3: float, V4: float, V5: float,
    V6: float, V7: float, V8: float, V9: float, V10: float,
    V11: float, V12: float, V13: float, V14: float, V15: float,
    V16: float, V17: float, V18: float, V19: float, V20: float,
    V21: float, V22: float, V23: float, V24: float, V25: float,
    V26: float, V27: float, V28: float, Amount: float,
) -> dict:
    """Explain WHY a transaction was flagged or passed, using SHAP values.

    Args:
        Same parameters as predict_tool.

    Returns:
        Dict with ensemble_score, is_fraud, and top_reasons (SHAP features).
    """
    tx = {
        "Time": Time,
        "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5,
        "V6": V6, "V7": V7, "V8": V8, "V9": V9, "V10": V10,
        "V11": V11, "V12": V12, "V13": V13, "V14": V14, "V15": V15,
        "V16": V16, "V17": V17, "V18": V18, "V19": V19, "V20": V20,
        "V21": V21, "V22": V22, "V23": V23, "V24": V24, "V25": V25,
        "V26": V26, "V27": V27, "V28": V28, "Amount": Amount,
    }
    return explain_transaction(tx, top_n=5)


# ---------- Tool registry (name -> callable) ----------

TOOL_REGISTRY = {
    "predict_tool": predict_tool,
    "explain_tool": explain_tool,
}


# ---------- Chat function ----------

SYSTEM_PROMPT = """You are SentinelFraud Assistant, an AI helping fraud analysts
understand credit card transactions.

You have two tools:
1. predict_tool — score a transaction for fraud.
2. explain_tool — explain WHY a transaction was scored that way (SHAP-based reasons).

When the user provides transaction details (Time, V1-V28, Amount), call predict_tool.
If they ask "why" or want reasons, call explain_tool.

RULES:
- Never invent fraud scores. Only report what the tools return.
- If the user hasn't provided all 30 fields (Time, V1-V28, Amount), ask for the missing ones.
- Keep answers concise and analyst-friendly.
- If a transaction is fraud (is_fraud=True), also call explain_tool to provide reasons.
"""


def chat(message: str) -> dict:
    """
    Main chat entry point.

    Sends the user message + tool definitions to Gemini. If Gemini decides
    to call a tool, we execute it locally and send the result back to Gemini
    for a final natural-language response.

    Returns:
        {
          "reply": str,              # final natural-language answer
          "tool_calls": [...],       # which tools were called (for debugging)
          "raw_tool_results": [...]  # raw outputs from predict/explain
        }
    """
    client = get_client()

    # Build tool declarations Gemini can use
    tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="predict_tool",
                description=predict_tool.__doc__,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "Time": types.Schema(type=types.Type.NUMBER),
                        "Amount": types.Schema(type=types.Type.NUMBER),
                        **{f"V{i}": types.Schema(type=types.Type.NUMBER) for i in range(1, 29)},
                    },
                    required=["Time", "Amount"] + [f"V{i}" for i in range(1, 29)],
                ),
            ),
            types.FunctionDeclaration(
                name="explain_tool",
                description=explain_tool.__doc__,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "Time": types.Schema(type=types.Type.NUMBER),
                        "Amount": types.Schema(type=types.Type.NUMBER),
                        **{f"V{i}": types.Schema(type=types.Type.NUMBER) for i in range(1, 29)},
                    },
                    required=["Time", "Amount"] + [f"V{i}" for i in range(1, 29)],
                ),
            ),
        ])
    ]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=tools,
        temperature=0.2,
    )

    # First turn: user message
    contents = [types.Content(role="user", parts=[types.Part(text=message)])]

    tool_calls = []
    tool_results = []

    # Loop — allow up to 3 tool-call rounds
    for _ in range(3):
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=contents,
            config=config,
        )

        # Check if the model wants to call a tool
        candidate = response.candidates[0]
        function_calls = [
            part.function_call
            for part in candidate.content.parts
            if hasattr(part, "function_call") and part.function_call
        ]

        if not function_calls:
            # No more tool calls — return the final text
            return {
                "reply": response.text or "(no response)",
                "tool_calls": tool_calls,
                "raw_tool_results": tool_results,
            }

        # Execute each tool call
        contents.append(candidate.content)  # add model's tool-call turn
        tool_response_parts = []

        for fc in function_calls:
            fn_name = fc.name
            fn_args = dict(fc.args)
            tool_calls.append({"name": fn_name, "args": fn_args})

            if fn_name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[fn_name](**fn_args)
                tool_results.append({"name": fn_name, "result": result})
                tool_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fn_name,
                            response=result,
                        )
                    )
                )
            else:
                tool_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fn_name,
                            response={"error": f"Unknown tool: {fn_name}"},
                        )
                    )
                )

        contents.append(types.Content(role="user", parts=tool_response_parts))

    return {
        "reply": "Max tool-call rounds reached without final response.",
        "tool_calls": tool_calls,
        "raw_tool_results": tool_results,
    }