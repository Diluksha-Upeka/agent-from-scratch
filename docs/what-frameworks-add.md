# What Agent Frameworks Add (LlamaIndex vs. Raw Loop)

When building an agent from scratch (like our `chat.py`), you write the raw "Reason → Act → Observe" loop yourself using the LLM provider's API. 

However, production frameworks like **LlamaIndex** or **LangChain** abstract this loop into dedicated classes (e.g., LlamaIndex's `AgentRunner` and `AgentWorker`). 

Here is a breakdown of what these frameworks provide out-of-the-box compared to a raw from-scratch implementation:

## The Core Difference: Runner vs Worker Architecture

LlamaIndex splits the agent loop into two pieces:
1. **`AgentWorker` (The Executor):** Holds the logic for executing a *single step*. It takes the current state, calls the LLM, decides if a tool should be called, executes the tool, and returns the result.
2. **`AgentRunner` (The Orchestrator):** Manages the overall *Task*. It maintains the conversation memory, handles the `while` loop, tracks the state across multiple steps, and returns the final response to the user.

## Feature Comparison Table

| Feature | Our Raw Loop (`chat.py`) | LlamaIndex (`AgentRunner`) |
|---------|--------------------------|----------------------------|
| **Tool Definition** | Manual JSON schemas (`types.Tool`) and a string-to-function dictionary registry. | Auto-generates schemas directly from Python type hints and docstrings (`FunctionTool.from_defaults`). |
| **State Management** | A single `messages` Python list passed back and forth. | Managed `Task` objects that can be paused, inspected mid-loop, and resumed. |
| **Conversational Memory** | Hardcoded JSON file (`agent_memory.json`) saving flat strings. | Built-in `ChatMemoryBuffer` with token-window sliding and summarization. |
| **Error Handling & Retries** | Manual `try/except` wrapping every tool, plus a manual loop over fallback models. | Built-in exponential backoff, automatic retry prompts when the LLM hallucinates tool args, and central error policies. |
| **Observability (Callbacks)** | Manual `print()` statements for tokens and tool execution. | Rich Event/Callback system (`on_tool_start`, `on_llm_end`) that hooks into tracing platforms (e.g., LangSmith, Arize). |
| **Streaming** | None. Waits for the entire string before printing. | Built-in token-by-token streaming back to the user, even *during* tool execution. |
| **Structured Output** | Relies entirely on the LLM adhering to the JSON schema. | First-class Pydantic integration. Framework automatically retries or repairs invalid JSON. |
| **Loop Control** | Hardcoded `MAX_ITERATIONS = 10` counter. | Advanced stopping conditions, timeout limits, and human-in-the-loop approval steps. |

## Why use a Framework?
- **Speed to market:** You don't have to write error-handling, token-sliding, or tool-parsing logic.
- **Observability:** Hooking into tracing tools is a one-liner.
- **Ecosystem:** You get pre-built tools (e.g., "Neo4j Query Tool", "Web Search Tool") instead of writing them from scratch.

## Why build from Scratch?
- **Complete control:** Frameworks often obscure *why* a loop failed or *exactly* what prompt was sent.
- **Minimal dependencies:** No framework bloat, meaning fewer dependency conflicts and a smaller Docker image.
- **Cost optimization:** You have total control over exactly which tokens are sent on every turn, allowing you to heavily optimize latency and cost.
