# Coder Agent

A GenAI agent that helps with coding tasks.

This is a fork of the coder_agent in the same project, but adapted to use Gemini SDK instead of OpenAI. The reason is that Gemini has a free tier for API access.

## Requirements

- Python 3.13 or higher
- Gemini API key. Obtain a free one here [https://aistudio.google.com/]

## Installation

1. Install uv (Python package manager):

   ```bash
   # Or install from the web
   curl -sSf https://install.python-uv.org | python3
   ```

2. Set up your Gemeni API key:
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   # On Windows terminal: set GEMINI_API_KEY=your_api_key_here
   # On Windows Powershell: $env:GEMINI_API_KEY=your_api_key_here
   ```

## Running the Agent

To start the coder agent, run:

```bash
uv run coder_agent.py
```