"""Agent loop tests with a scripted fake LLM (no API key, no network)."""
import json

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent import analytics as A
from agent.agent import SensorAgent
from agent.knowledge import KnowledgeBase, build_documents


class ScriptedLLM:
    """Returns pre-defined AIMessages in order and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def make_agent(replies, max_steps=5):
    df = A.load_data()
    kb = KnowledgeBase(DeterministicFakeEmbedding(size=32), build_documents(df))
    llm = ScriptedLLM(replies)
    return SensorAgent(llm, df, kb, max_steps=max_steps), llm


def call(name, args, cid="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def test_tool_schemas_are_valid_for_function_calling():
    agent, llm = make_agent([AIMessage(content="ok")])
    names = {convert_to_openai_tool(t)["function"]["name"] for t in llm.bound_tools}
    assert names == {"dataset_overview", "count_machines", "failure_rate", "sensor_statistic",
                     "count_threshold", "compare_failed_vs_normal", "search_documents"}
    schema = convert_to_openai_tool(agent.by_name["sensor_statistic"])["function"]["parameters"]
    assert "torque" in json.dumps(schema)                      # enum of sensors is exposed


def test_tool_result_is_sent_back_and_answer_returned():
    agent, llm = make_agent([call("count_machines", {"failed": True}),
                             AIMessage(content="339 machines failed.")])
    res = agent.ask("How many machines failed?")
    assert res.answer == "339 machines failed."
    assert res.steps[0].tool == "count_machines"
    assert res.steps[0].result["count"] == 339
    tool_msg = llm.seen[1][-1]
    assert isinstance(tool_msg, ToolMessage) and '"count": 339' in tool_msg.content


def test_invalid_arguments_return_an_error_to_the_model():
    agent, _ = make_agent([call("sensor_statistic", {"sensor": "voltage"}),
                           AIMessage(content="That sensor does not exist.")])
    res = agent.ask("Average voltage?")
    assert "error" in res.steps[0].result


def test_unknown_tool_is_reported():
    agent, _ = make_agent([call("delete_everything", {}), AIMessage(content="Sorry.")])
    res = agent.ask("x")
    assert res.steps[0].result == {"error": "unknown tool 'delete_everything'"}


def test_search_tool_returns_documents():
    agent, _ = make_agent([call("search_documents", {"query": "heat dissipation"}),
                           AIMessage(content="...")])
    res = agent.ask("What is HDF?")
    assert len(res.steps[0].result["results"]) == 3


def test_step_limit_forces_a_final_answer():
    loop = [call("dataset_overview", {}, cid=f"c{i}") for i in range(3)]
    agent, llm = make_agent(loop + [AIMessage(content="final")], max_steps=3)
    res = agent.ask("loop forever")
    assert res.answer == "final"
    assert len(res.steps) == 3
