import os
import json
import subprocess
from rich import print
from google.genai import Client
from google.genai import types

client = Client()

agent_state = {"messages": []}
SYSTEM_INSTRUCTION = """
You are a helpful coding assistant. Your goal is to help the user with programming tasks.
            
You have access to the following tools:
1. edit_file: Modify files by replacing text or create new files
2. run_command: Execute shell commands
3. list_directory: View the contents of directories
4. read_file_content: Read the content of files

For each user request:
1. Understand what the user is trying to accomplish
2. Break down complex tasks into smaller steps
3. Use your tools to gather information about the codebase when needed
4. Implement solutions by writing or modifying code
5. Explain your reasoning and approach

When modifying code, be careful to maintain the existing style and structure. Test your changes when possible.
If you're unsure about something, ask clarifying questions before proceeding.

You must run and test your changes before reporting success.""".strip()


def edit_file(filename: str, find_str: str, replace_str: str) -> bool:
    # If the file doesn't exist and find_str is empty, create the file
    if not os.path.exists(filename) and find_str == "":
        with open(filename, "w") as f:
            f.write(replace_str)
        return True

    # If the file exists, read it and replace the string
    try:
        with open(filename, "r") as f:
            content = f.read()

        # Replace the string
        if find_str in content:
            new_content = content.replace(find_str, replace_str)

            # Write the modified content back to the file
            with open(filename, "w") as f:
                f.write(new_content)
            return True
    except FileNotFoundError:
        print(f"File {filename} not found and find_str is not empty.")

    return False


def run_command(command, working_dir) -> tuple[str, int]:
    try:
        # Set up the process with the working directory if provided
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=working_dir,
        )

        # Get the output and return code
        output, _ = process.communicate()
        error_code = process.returncode

        # Clip output to first 1000 and last 1000 characters if it's longer than 2000 characters
        if len(output) > 2000:
            output = output[:1000] + "\n\n[...content clipped...]\n\n" + output[-1000:]

        return output, error_code
    except Exception as e:
        return str(e), 1


def list_directory(path=".") -> str:
    try:
        # Get the list of files and directories
        items = os.listdir(path)

        # Format the output
        if not items:
            return f"Directory '{path}' is empty."

        result = f"Contents of directory '{path}':\n"
        for item in items:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                item_type = "Directory"
            else:
                item_type = "File"
            result += f"- {item} ({item_type})\n"

        return result.strip()
    except FileNotFoundError:
        return f"Error: Directory '{path}' not found."
    except PermissionError:
        return f"Error: Permission denied to access '{path}'."
    except Exception as e:
        return f"Error listing directory '{path}': {str(e)}"


def read_file_content(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            if len(content) > 2000:
                # Get first 1000 and last 1000 characters
                first_part = content[:1000]
                last_part = content[-1000:]
                content = first_part + "\n\n[...content clipped...]\n\n" + last_part
        return content
    except FileNotFoundError:
        return f"Error: File '{path}' not found."
    except PermissionError:
        return f"Error: Permission denied to access '{path}'."
    except UnicodeDecodeError:
        return f"Error: Unable to decode '{path}'. The file might be binary or use an unsupported encoding."
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"


TOOLS_DECL = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="edit_file",
                description="Apply a diff to a file by replacing occurrences of find_str with replace_str.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "filename": types.Schema(
                            type="STRING", description="The name of the file to modify"
                        ),
                        "find_str": types.Schema(
                            type="STRING", description="The string to find in the file"
                        ),
                        "replace_str": types.Schema(
                            type="STRING", description="The string to replace with"
                        ),
                    },
                    required=["filename", "find_str", "replace_str"],
                ),
            )
        ]
    ),
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="run_command",
                description="Run a shell command and return its output and error code.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "command": types.Schema(
                            type="STRING", description="The command to run in the shell"
                        ),
                        "working_dir": types.Schema(
                            type="STRING",
                            description="The working directory to run the command in",
                        ),
                    },
                    required=["command", "working_dir"],
                ),
            )
        ]
    ),
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="list_directory",
                description="List the contents of a directory.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "path": types.Schema(
                            type="STRING",
                            description="The path to the directory to list. Defaults to the current directory.",
                        ),
                    },
                    required=["path"],
                ),
            )
        ]
    ),
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="read_file_content",
                description="Read and return the content of a file.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "path": types.Schema(
                            type="STRING",
                            description="The path to the file to read. Defaults to the current directory.",
                        ),
                    },
                    required=["path"],
                ),
            )
        ]
    ),
]


# exit if last message doesn't have any tool calls (agent doesn't want to take any more actions)
def is_goal_achieved(state) -> bool:
    if len(state["messages"]) <= 2:
        return False

    last_message = state["messages"][-1]

    return (
        "role" in last_message
        and last_message["role"] == "assistant"
        and "tool_calls" not in last_message
    )


def ask_user_approval(message: str) -> bool:
    user_approval = input(f"{message} (y/n): ")
    return user_approval.lower() == "y"


def loop(user_input: str):
    user_message = types.Content(
        role="user", parts=[types.Part.from_text(text=user_input)]
    )
    agent_state["messages"].append(user_message)

    while not is_goal_achieved(agent_state) and len(agent_state["messages"]) < 100:
        print(f"[Thinking... step {len(agent_state['messages']) - 1}]")
        completion = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=agent_state["messages"],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION, tools=TOOLS_DECL
            ),
        )
        print(completion)
        if completion.candidates == None:
            return

        agent_state["messages"].append(completion.candidates[0].content)
        tool_result_content = types.Content(role="tool", parts=[])
        # each required function call will be in a separate part.
        for part in completion.candidates[0].content.parts:
            if part.function_call:
                tool_call = part.function_call
                print(tool_call)
                arguments = tool_call.args

                if tool_call.name == "edit_file":
                    print(f"""Editing file: {arguments["filename"]}""")
                    if arguments["find_str"] != "":
                        print(f"""Content to find\n```\n{arguments["find_str"]}\n```""")
                    if arguments["replace_str"] != "":
                        print(
                            f"""Content to replace with\n```\n{arguments["replace_str"]}\n```"""
                        )
                    if not ask_user_approval("Do you want to edit this file?"):
                        print("File edit cancelled by user.")
                        result = ("File edit cancelled by user.", 0)
                        continue
                    result = edit_file(**arguments)
                    print(result)

                elif tool_call.name == "run_command":
                    print(f"""Executing command: {arguments["command"]}""")
                    # Ask for user approval before executing the command
                    if not ask_user_approval("Do you want to execute this command?"):
                        print("Command execution cancelled by user.")
                        result = ("Command execution cancelled by user.", 0)
                        continue
                    result = run_command(**arguments)
                    print(result[0])

                elif tool_call.name == "list_directory":
                    print(f"""Listing directory: {arguments["path"]}""")
                    result = list_directory(**arguments)
                    print(result)

                elif tool_call.name == "read_file_content":
                    print(f"""Reading file: {arguments["path"]}""")
                    result = read_file_content(**arguments)
                    print(result)

                tool_result_content.parts.append(
                    types.Part.from_function_response(
                        name=tool_call.name,  # Must match the name the model requested
                        response={
                            "result": str(result)
                        },  # The actual result of the function call
                    )
                )

        if tool_result_content.parts != []:
            agent_state["messages"].append(tool_result_content)


if __name__ == "__main__":
    user_input = input("How can I help you?\n")
    loop(user_input)
