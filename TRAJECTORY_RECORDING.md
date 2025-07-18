# Trajectory Recording Functionality

This document describes the trajectory recording functionality added to the Trae Agent project. The system captures detailed information about LLM interactions and agent execution steps for analysis, debugging, and auditing purposes.

## Overview

The trajectory recording system captures:
- **Raw LLM interactions**: Input messages, responses, token usage, and tool calls for both Anthropic and OpenAI clients
- **Agent execution steps**: State transitions, tool calls, tool results, reflections, and errors
- **Metadata**: Task description, timestamps, model configuration, and execution metrics

## Key Components

### 1. TrajectoryRecorder (`my_agent/utils/trajectory_recorder.py`)

The `TrajectoryRecorder` class is the core component responsible for managing trajectory data. It handles the creation, updating, and saving of trajectory files in JSON format.

**Key Methods:**
- `start_recording()`: Initializes the recording process with task details.
- `record_llm_interaction()`: Logs all messages and responses exchanged with the LLM.
- `record_agent_step()`: Captures the agent's state transitions and actions.
- `finalize_recording()`: Completes the recording and saves the trajectory file.

### 2. LLM Client Integration

Trajectory recording is integrated directly into the LLM clients to automatically capture all interactions.

**Anthropic Client** (`my_agent/utils/anthropic_client.py`):
- The `chat()` method calls `self.trajectory_recorder.record_llm_interaction()` after receiving a response from the Anthropic API.

**OpenAI Client** (`my_agent/utils/openai_client.py`):
- Similarly, the `chat()` method in the OpenAI client logs interactions using the trajectory recorder.

### 3. Agent Integration

The `Agent` class and its subclasses use the `TrajectoryRecorder` to log higher-level agent activities.

- **`Agent.set_trajectory_recorder()`**: Assigns a recorder instance to the agent and its LLM client.
- **`Agent.execute_task()`**: Calls `record_agent_step()` to log each step of the task execution.

## Usage

### CLI Usage

#### Basic Recording (Auto-generated filename)
```bash
my-agent-cli run "Create a hello world Python script"
# Trajectory saved to: trajectory_20250612_220546.json
```

#### Custom Filename
```bash
my-agent-cli run "Fix the bug in main.py" --trajectory-file my_debug_session.json
# Trajectory saved to: my_debug_session.json
```

#### Interactive Mode
```bash
my-agent-cli interactive --trajectory-file session.json
```

### Programmatic Usage

```python
from my_agent.agent.my_agent import MyAgent
from my_agent.utils.llm_client import LLMProvider
from my_agent.utils.config import ModelParameters

# Create agent
agent = MyAgent(LLMProvider.ANTHROPIC, model_parameters, max_steps=10)

# Set up trajectory recording
trajectory_path = agent.setup_trajectory_recording("my_trajectory.json")

# Configure and run task
agent.new_task("My task", task_args)
execution = await agent.execute_task()

# Trajectory is automatically saved
print(f"Trajectory saved to: {trajectory_path}")
```

## Trajectory File Format

The trajectory file is a JSON document with the following structure:

```json
{
  "task": "Description of the task",
  "start_time": "2025-06-12T22:05:46.433797",
  "end_time": "2025-06-12T22:06:15.123456",
  "provider": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "max_steps": 20,
  "llm_interactions": [
    {
      "timestamp": "2025-06-12T22:05:47.000000",
      "provider": "anthropic",
      "model": "claude-sonnet-4-20250514",
      "input_messages": [
        {
          "role": "system",
          "content": "You are a software engineering assistant..."
        },
        {
          "role": "user",
          "content": "Create a hello world Python script"
        }
      ],
      "response": {
        "content": "I'll help you create a hello world Python script...",
        "model": "claude-sonnet-4-20250514",
        "finish_reason": "end_turn",
        "usage": {
          "input_tokens": 150,
          "output_tokens": 75,
          "cache_creation_input_tokens": 0,
          "cache_read_input_tokens": 0,
          "reasoning_tokens": null
        },
        "tool_calls": [
          {
            "call_id": "call_123",
            "name": "str_replace_based_edit_tool",
            "arguments": {
              "command": "create",
              "path": "hello.py",
              "file_text": "print('Hello, World!')"
            }
          }
        ]
      },
      "tools_available": ["str_replace_based_edit_tool", "bash", "task_done"]
    }
  ],
  "agent_steps": [
    {
      "step_number": 1,
      "timestamp": "2025-06-12T22:05:47.500000",
      "state": "thinking",
      "llm_messages": [...],
      "llm_response": {...},
      "tool_calls": [
        {
          "call_id": "call_123",
          "name": "str_replace_based_edit_tool",
          "arguments": {...}
        }
      ],
      "tool_results": [
        {
          "call_id": "call_123",
          "success": true,
          "result": "File created successfully",
          "error": null
        }
      ],
      "reflection": null,
      "error": null
    }
  ],
  "success": true,
  "final_result": "Hello world Python script created successfully!",
  "execution_time": 28.689999
}
```

### Field Descriptions

**Root Level:**
- `task`: The original task description
- `start_time`/`end_time`: ISO format timestamps
- `provider`: LLM provider used ("anthropic" or "openai")
- `model`: Model name
- `max_steps`: Maximum allowed execution steps
- `success`: Whether the task completed successfully
- `final_result`: Final output or result message
- `execution_time`: Total execution time in seconds

**LLM Interactions:**
- `timestamp`: When the interaction occurred
- `input_messages`: Messages sent to the LLM
- `response`: Complete LLM response including content, usage, and tool calls
- `tools_available`: List of tools available during this interaction

**Agent Steps:**
- `step_number`: Sequential step number
- `state`: Agent state ("thinking", "calling_tool", "reflecting", "completed", "error")
- `llm_messages`: Messages used in this step
- `llm_response`: LLM response for this step
- `tool_calls`: Tools called in this step
- `tool_results`: Results from tool execution
- `reflection`: Agent's reflection on the step
- `error`: Error message if the step failed

## Benefits

1. **Debugging**: Trace exactly what happened during agent execution
2. **Analysis**: Understand LLM reasoning and tool usage patterns
3. **Auditing**: Maintain records of what changes were made and why
4. **Research**: Analyze agent behavior for improvements
5. **Compliance**: Keep detailed logs of automated actions

## File Management

- Trajectory files are saved in the current working directory by default
- Files use timestamp-based naming if no custom path is provided
- Files are automatically created/overwritten
- The system handles directory creation if needed
- Files are saved continuously during execution (not just at the end)

## Security Considerations

- Trajectory files may contain sensitive information (API keys are not logged)
- Store trajectory files securely if they contain proprietary code or data
- Consider excluding trajectory files from version control (add `trajectory_*.json` to `.gitignore`)

## Example Use Cases

1. **Debugging Failed Tasks**: Review what went wrong in agent execution
2. **Performance Analysis**: Analyze token usage and execution patterns
3. **Compliance Auditing**: Track all changes made by the agent
4. **Model Comparison**: Compare behavior across different LLM providers/models
5. **Tool Usage Analysis**: Understand which tools are used and how often
