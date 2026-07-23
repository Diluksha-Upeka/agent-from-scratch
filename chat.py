import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pathlib import Path

TEST_DATA_DIR = Path("test_data").resolve()

# Load environment variables
load_dotenv()

# Initialize the client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            initial_delay=2.0,       # Start with a 2-second sleep
            attempts=6,              # Retry up to 6 times before failing
            http_status_codes=[429, 500, 503, 504], # Explicitly catch 503
        ),
        timeout=60 * 1000,           # 60-second connection timeout
    )
)

MODELS = [
    'gemini-3.6-flash',       # newest, confirmed working
    'gemini-3.5-flash-lite',  # confirmed working
    'gemini-3.1-flash-lite',  # confirmed working
]
REQUEST_TIMEOUT = 15_000  # 15 seconds per request
MAX_TOOL_RESULT_LENGTH = 5000  # ~1250 tokens — truncate tool results beyond this

# 1. Tool Implementation

def calculator(expression: str) -> str:
    """Evaluate a math expression string and return the result."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def read_file(path: str) -> str:
    """Read and return contents of a file from the test_data folder."""

    try:
        if "crash" in path.lower():
            raise RuntimeError("Intentional crash for testing!")

        # 1. Resolve path relative to TEST_DATA_DIR
        file_path = (TEST_DATA_DIR / path).resolve()

        # 2. Security: ensure file is inside TEST_DATA_DIR
        if TEST_DATA_DIR not in file_path.parents:
            return "Error: Access denied. File is outside test_data directory."

        # 3. Read file contents
        if not file_path.exists():
            return f"Error: File '{path}' not found."

        if not file_path.is_file():
            return f"Error: '{path}' is not a file."

        return file_path.read_text(encoding="utf-8")

    except Exception as e:
        return f"Error reading file: {str(e)}"  

def search_files(pattern: str) -> str:
    """Search filenames and file contents matching a pattern in test_data."""

    results = []
    pattern_lower = pattern.lower()

    try:
        # 1. Walk through TEST_DATA_DIR
        for file_path in TEST_DATA_DIR.iterdir():

            # Only process files
            if not file_path.is_file():
                continue

            filename = file_path.name

            # 2. Check filename match
            if pattern_lower in filename.lower():
                results.append(
                    f"Found in {filename}: filename match"
                )

            # Check file contents
            try:
                content = file_path.read_text(encoding="utf-8")

                for line in content.splitlines():
                    if pattern_lower in line.lower():
                        # 3. Add matching line
                        results.append(
                            f"Found in {filename}: {line.strip()}"
                        )

            except UnicodeDecodeError:
                continue

        if not results:
            return f"No matches found for '{pattern}'."

        return "\n".join(results)

    except Exception as e:
        return f"Error searching files: {str(e)}"


# Registry: tool name → Python function
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "read_file": read_file,
    "search_files": search_files,
}

# 2. Tool Declaration (Gemini function-calling format)

calculator_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="calculator",
            description=(
                "Evaluate a mathematical expression. "
                "Use this for ANY arithmetic calculation the user asks about."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "expression": types.Schema(
                        type="STRING",
                        description=(
                            "A valid Python math expression, "
                            "e.g. '247 * 38' or '3 * 12.50 * 1.08'"
                        ),
                    ),
                },
                required=["expression"],
            ),
        )
    ]
)

read_file_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="read_file",
            description=(
                "Read and return contents of a file from the test_data folder. "
                "Use this for ANY file reading the user asks about."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "path": types.Schema(
                        type="STRING",
                        description=(
                            "A valid file path, "
                            "e.g. 'recipe.md' or 'contacts.csv'"
                        ),
                    ),
                },
                required=["path"],
            ),
        )
    ]
)

search_files_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_files",
            description=(
                "Search filenames and file contents matching a pattern in test_data. "
                "Use this for ANY file search the user asks about."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "pattern": types.Schema(
                        type="STRING",
                        description=(
                            "A search pattern, "
                            "e.g. 'recipe' or 'contacts'"
                        ),
                    ),
                },
                required=["pattern"],
            ),
        )
    ]
)

# 3. Helpers

def has_function_calls(response) -> bool:
    """Check if the model response contains any function call parts."""
    for part in response.candidates[0].content.parts:
        if part.function_call:
            return True
    return False


def call_model(messages):
    """Try each model in MODELS with a per-request timeout. Falls back on failure."""
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=messages,
                config=types.GenerateContentConfig(
                    tools=[calculator_tool, read_file_tool, search_files_tool],
                    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT),
                ),
            )
            return response
        except Exception as e:
            print(f"[WARN] {model} failed ({type(e).__name__}), trying next...")
            print(f"Error details: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()   # <- full stack trace for debugging
            print(f"Error type: {type(e)}")
            
    raise RuntimeError("All models failed. Check your API key and quota.")


# 4. Agentic Loop with Iteration Guard & Token Tracking

MAX_ITERATIONS = 10

messages = []

print("Chat Session started (with tool calling!)")
print("Type 'exit' to quit.\n" + "=" * 60 + "\n")

while True:
    user_text = input("You: ")
    if user_text.lower() in ["exit", "quit", "q", "bye"]:
        print("\nChat ended. Goodbye!")
        break

    if not user_text.strip():
        continue

    print(f"\n[USER]       {user_text}")

    # Append the user message to history
    messages.append(
        types.Content(role="user", parts=[types.Part(text=user_text)])
    )

    # Initialize iteration counter and token tracker
    iteration = 0               # iteration counter
    total_input_tokens = 0      # input token tracker
    total_output_tokens = 0     # output token tracker
    last_response_text = None   # last response text

    # Agentic loop
    while iteration < MAX_ITERATIONS:
        # Call model and log tokens
        response = call_model(messages)

        # Extract and accumulate token usage
        usage = response.usage_metadata
        turn_input = usage.prompt_token_count or 0
        turn_output = usage.candidates_token_count or 0
        total_input_tokens += turn_input
        total_output_tokens += turn_output
        print(
            f"  [TOKENS] Turn {iteration}: "
            f"input={turn_input}, output={turn_output} | "
            f"Running total: input={total_input_tokens}, output={total_output_tokens}"
        )

        # Evaluate — if no tool calls, we have the final answer
        if not has_function_calls(response):
            last_response_text = response.text
            print(f"[MODEL] {last_response_text}\n")
            # Save final text turn into history
            messages.append(response.candidates[0].content)
            break

        # Execute tools and append results
        # Save the model's tool-call turn into history
        messages.append(response.candidates[0].content)

        function_response_parts = []
        for part in response.candidates[0].content.parts:
            if part.function_call:
                fn_name = part.function_call.name
                fn_args = dict(part.function_call.args)

                args_str = ", ".join(f'{k}="{v}"' for k, v in fn_args.items())
                print(f"[MODEL] Tool call: {fn_name}({args_str})")

                # Dispatch to the actual Python function
                fn = TOOL_FUNCTIONS.get(fn_name)
                if fn:
                    try:
                        result = fn(**fn_args)
                    except Exception as e:
                        result = f"Error executing {fn_name}: {type(e).__name__}: {e}"
                else:
                    result = f"Error: unknown tool '{fn_name}'"

                # Scenario C: Type validation — ensure result is always a string
                if not isinstance(result, str):
                    result = f"Error: tool '{fn_name}' returned {type(result).__name__} instead of str. Value: {result}"

                # Scenario B: Truncate huge results to prevent context bloat
                if len(result) > MAX_TOOL_RESULT_LENGTH:
                    original_len = len(result)
                    result = result[:MAX_TOOL_RESULT_LENGTH] + (
                        f"\n... [TRUNCATED — original was {original_len} chars]"
                    )
                    print(f"[TOOL] Result truncated: {original_len} -> {MAX_TOOL_RESULT_LENGTH} chars")

                print(f"[TOOL] Result: {result}")

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": result},
                    )
                )

        # Send tool results back to the model
        messages.append(
            types.Content(role="user", parts=function_response_parts)
        )

        # Safety guard - prevent infinite loops
        iteration += 1
        if iteration >= MAX_ITERATIONS:
            print(
                f"  [SAFETY] Max iterations ({MAX_ITERATIONS}) reached! "
                f"Forcing stop to prevent runaway loop."
            )
            # Preserve whatever the model last generated
            last_response_text = (
                response.text if response.text
                else "[Agent stopped: max iterations exceeded]"
            )
            break

    # Per-turn summary
    print(
        f"  [SUMMARY] Turn finished in {iteration + 1} iteration(s) — "
        f"Total tokens: input={total_input_tokens}, output={total_output_tokens}\n"
    )
