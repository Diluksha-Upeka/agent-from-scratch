import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

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

# 1. Tool Implementation

def calculator(expression: str) -> str:
    """Evaluate a math expression string and return the result."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Registry: tool name → Python function
TOOL_FUNCTIONS = {
    "calculator": calculator,
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
                    tools=[calculator_tool],
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
                    result = fn(**fn_args)
                else:
                    result = f"Error: unknown tool '{fn_name}'"

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
