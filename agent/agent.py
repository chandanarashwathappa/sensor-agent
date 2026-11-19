"""Tool-calling agent: the LLM chooses which tool answers a question.

    numbers (counts, averages, rates, thresholds) -> analytics tools (exact pandas)
    definitions, units, failure causes             -> search_documents (FAISS RAG)

Every tool call and its result is recorded, so the UI can show exactly how
an answer was produced.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from . import analytics as A
from .knowledge import KnowledgeBase

SYSTEM_PROMPT = """You answer questions about the AI4I 2020 predictive-maintenance dataset \
(10,000 milling-machine records, 339 failures).

Rules:
- For ANY number (count, average, rate, minimum, maximum, threshold question) call an analytics tool. \
Never estimate or invent numbers.
- For definitions, units and how failure modes are caused, call search_documents.
- Temperatures are in K, rotational speed in rpm, torque in Nm, tool wear in minutes, power in W.
- If the tools cannot answer the question (for example dates, costs or other machines), say that the \
dataset does not contain that information.
- Answer briefly and include the numbers with units."""

SensorName = Literal[tuple(A.SENSORS)]
ProductType = Literal["L", "M", "H"]
FailureMode = Literal["TWF", "HDF", "PWF", "OSF", "RNF"]


class CountArgs(BaseModel):
    failed: Optional[bool] = Field(None, description="True = only failed machines, False = only machines without failure")
    product_type: Optional[ProductType] = Field(None, description="product quality type")
    failure_mode: Optional[FailureMode] = Field(None, description="only machines with this failure mode")


class RateArgs(BaseModel):
    product_type: Optional[ProductType] = Field(None, description="product type; omit for all machines")


class StatArgs(CountArgs):
    sensor: SensorName = Field(..., description="sensor to summarise")
    statistic: Literal["mean", "median", "min", "max", "std"] = Field("mean")


class ThresholdArgs(BaseModel):
    sensor: SensorName
    comparison: Literal[">", ">=", "<", "<="]
    value: float = Field(..., description="threshold in the sensor's unit")
    failed: Optional[bool] = None
    product_type: Optional[ProductType] = None


class CompareArgs(BaseModel):
    sensor: SensorName


class SearchArgs(BaseModel):
    query: str = Field(..., description="what to look up in the documentation")


class NoArgs(BaseModel):
    pass


@dataclass
class Step:
    tool: str
    args: dict
    result: object


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)


def _text(msg: AIMessage) -> str:
    c = msg.content
    if isinstance(c, list):                      # some providers return content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return str(c)


class SensorAgent:
    def __init__(self, llm, df: pd.DataFrame, kb: KnowledgeBase, max_steps: int = 5):
        self.df = df
        self.kb = kb
        self.max_steps = max_steps
        self.tools = self._make_tools()
        self.by_name = {t.name: t for t in self.tools}
        self.llm = llm
        self.llm_with_tools = llm.bind_tools(self.tools)

    def _make_tools(self) -> list[StructuredTool]:
        df = self.df

        def search(query: str) -> dict:
            return {"results": [{"source": h.source, "text": h.text} for h in self.kb.search(query, k=3)]}

        def mk(fn, name, desc, schema):
            return StructuredTool.from_function(func=fn, name=name, description=desc, args_schema=schema)

        return [
            mk(lambda: A.dataset_overview(df), "dataset_overview",
               "Records, failures, overall failure rate, product-type mix and failure-mode counts.", NoArgs),
            mk(lambda **kw: A.count_machines(df, **kw), "count_machines",
               "Count records, optionally only failed / non-failed, one product type or one failure mode.", CountArgs),
            mk(lambda **kw: A.failure_rate(df, **kw), "failure_rate",
               "Failure rate in percent, overall or for one product type (also returns all types).", RateArgs),
            mk(lambda **kw: A.sensor_statistic(df, **kw), "sensor_statistic",
               "mean/median/min/max/std of a sensor, with optional filters.", StatArgs),
            mk(lambda **kw: A.count_threshold(df, **kw), "count_threshold",
               "How many records have a sensor above/below a value, and how many of them failed.", ThresholdArgs),
            mk(lambda **kw: A.compare_failed_vs_normal(df, **kw), "compare_failed_vs_normal",
               "Mean of a sensor for failed machines vs machines without failure.", CompareArgs),
            mk(search, "search_documents",
               "Search the dataset documentation: definitions, units, causes of failure modes.", SearchArgs),
        ]

    def run_tool(self, name: str, args: dict) -> object:
        tool = self.by_name.get(name)
        if tool is None:
            return {"error": f"unknown tool '{name}'"}
        try:
            return tool.invoke(args)
        except Exception as exc:                # invalid arguments go back to the model
            return {"error": str(exc)}

    def ask(self, question: str) -> AgentResult:
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]
        steps: list[Step] = []
        for _ in range(self.max_steps):
            ai = self.llm_with_tools.invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                return AgentResult(_text(ai), steps)
            for call in ai.tool_calls:
                result = self.run_tool(call["name"], call.get("args", {}))
                steps.append(Step(call["name"], call.get("args", {}), result))
                messages.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"]))
        # step limit reached: ask for a final answer without tools
        messages.append(HumanMessage("Answer now with the information you have."))
        return AgentResult(_text(self.llm.invoke(messages)), steps)


def build_default_agent():
    """Agent with Mistral models; needs MISTRAL_API_KEY in the environment (.env)."""
    from dotenv import load_dotenv
    from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

    from .knowledge import build_documents

    load_dotenv()
    df = A.load_data()
    kb = KnowledgeBase(MistralAIEmbeddings(model="mistral-embed"), build_documents(df))
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0, max_tokens=512)
    return SensorAgent(llm, df, kb)
