# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""CLI console for MyAgent interaction."""

import asyncio
import os
from typing import Optional, TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table
from rich.markdown import Markdown

from .config import load_config
from ..agent.agent_basics import AgentStep, AgentState

if TYPE_CHECKING:
    from ..agent.my_agent import MyAgent


class CLIConsole:
    """Console interface for MyAgent."""
    
    def __init__(self):
        self.console = Console()
        self.agent: Optional[MyAgent] = None
        self.current_step: Optional[AgentStep] = None
        self.live_display: Optional[Live] = None
    
    @staticmethod
    def run(task_string: str, config_file: Optional[str] = None, trajectory_file: Optional[str] = None):
        """Run the agent with a given task."""
        console = CLIConsole()
        asyncio.run(console._run_task(task_string, config_file, trajectory_file))
    
    @staticmethod
    def interactive(config_file: Optional[str] = None, trajectory_file: Optional[str] = None):
        """Run the agent in interactive mode."""
        console = CLIConsole()
        asyncio.run(console._interactive_mode(config_file, trajectory_file))
    
    async def _run_task(self, task_string: str, config_file: Optional[str] = None, trajectory_file: Optional[str] = None):
        """Execute a single task."""
        try:
            # Load configuration
            config = load_config(config_file)
            
            # Import MyAgent here to avoid circular import
            from ..agent.my_agent import MyAgent
            
            # Create agent
            self.agent = MyAgent(config)
            self.agent.set_cli_console(self)
            
            # Set up trajectory recording
            if trajectory_file or True:  # Always record trajectories
                trajectory_path = self.agent.setup_trajectory_recording(trajectory_file)
                self.console.print(f"[dim]Recording trajectory to: {trajectory_path}[/dim]")
            
            # Display task
            self.console.print(Panel(
                Text(task_string, style="bold blue"),
                title="Task",
                border_style="blue"
            ))
            
            # Set up the task
            project_path = os.getcwd()
            self.agent.new_task(
                task=task_string,
                extra_args={
                    "project_path": project_path,
                    "issue": task_string
                }
            )
            
            # Execute the task
            with Live(self._create_status_display(), refresh_per_second=4) as live:
                self.live_display = live
                execution = await self.agent.execute_task()
                self.live_display = None
            
            # Display final result
            if execution.success:
                self.console.print(Panel(
                    Markdown(execution.final_result or "Task completed successfully!"),
                    title="✅ Task Completed",
                    border_style="green"
                ))
            else:
                self.console.print(Panel(
                    "Task was not completed successfully.",
                    title="❌ Task Failed",
                    border_style="red"
                ))
            
            # Display token usage
            if execution.total_tokens:
                self.console.print(f"\n[dim]Token usage: {execution.total_tokens.total_tokens} total ({execution.total_tokens.prompt_tokens} prompt + {execution.total_tokens.completion_tokens} completion)[/dim]")
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Task interrupted by user[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Error: {e}[/red]")
    
    async def _interactive_mode(self, config_file: Optional[str] = None, trajectory_file: Optional[str] = None):
        """Run in interactive mode."""
        try:
            # Load configuration
            config = load_config(config_file)
            
            self.console.print(Panel(
                "Welcome to MyAgent Interactive Mode!\n\n"
                "Commands:\n"
                "• Type any task to execute it\n"
                "• 'status' - Show agent information\n"
                "• 'help' - Show this help\n"
                "• 'clear' - Clear screen\n"
                "• 'exit' or 'quit' - Exit interactive mode",
                title="MyAgent Interactive",
                border_style="cyan"
            ))
            
            while True:
                try:
                    # Get user input
                    user_input = self.console.input("\n[bold cyan]MyAgent>[/bold cyan] ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['exit', 'quit']:
                        self.console.print("[yellow]Goodbye![/yellow]")
                        break
                    elif user_input.lower() == 'help':
                        self.console.print(Panel(
                            "Available commands:\n\n"
                            "• Type any task description to execute it\n"
                            "• 'status' - Show current agent configuration\n"
                            "• 'clear' - Clear the screen\n"
                            "• 'exit' or 'quit' - Exit interactive mode",
                            title="Help",
                            border_style="blue"
                        ))
                    elif user_input.lower() == 'status':
                        self._show_status(config)
                    elif user_input.lower() == 'clear':
                        self.console.clear()
                    else:
                        # Execute the task
                        await self._run_task(user_input, config_file, trajectory_file)
                
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Use 'exit' or 'quit' to leave interactive mode[/yellow]")
                except EOFError:
                    break
        
        except Exception as e:
            self.console.print(f"[red]Error in interactive mode: {e}[/red]")
    
    def _show_status(self, config):
        """Show current agent status."""
        table = Table(title="Agent Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Provider", config.default_provider)
        table.add_row("Model", config.model_providers[config.default_provider].model)
        table.add_row("Max Steps", str(config.max_steps))
        table.add_row("Lakeview", "Enabled" if config.enable_lakeview else "Disabled")
        
        self.console.print(table)
    
    def update_status(self, step: Optional["AgentStep"] = None, agent_execution=None):
        """Update the status display with current step information."""
        if step:
            self.current_step = step
        if self.live_display:
            self.live_display.update(self._create_status_display())
    
    def _create_status_display(self):
        """Create the status display panel."""
        if not self.current_step:
            return Panel("Initializing...", title="Status", border_style="blue")
        
        step = self.current_step
        
        # Create status content
        content = f"Step {step.step_number}: {self._get_state_emoji(step.state)} {step.state.value}"
        
        if step.state == AgentState.THINKING and step.llm_response:
            if step.llm_response.content:
                # Show first few lines of LLM response
                lines = step.llm_response.content.split('\n')[:3]
                preview = '\n'.join(lines)
                if len(lines) == 3 and len(step.llm_response.content.split('\n')) > 3:
                    preview += "\n..."
                content += f"\n\n[dim]{preview}[/dim]"
        
        elif step.state == AgentState.CALLING_TOOL and step.tool_calls:
            tool_names = [tc.name for tc in step.tool_calls]
            content += f"\n\nCalling tools: {', '.join(tool_names)}"
        
        elif step.state == AgentState.CALLING_TOOL and step.tool_results:
            content += f"\n\nTool results received ({len(step.tool_results)} results)"
        
        elif step.state == AgentState.REFLECTING and step.reflection:
            # Show first few lines of reflection
            lines = step.reflection.split('\n')[:2]
            preview = '\n'.join(lines)
            if len(lines) == 2 and len(step.reflection.split('\n')) > 2:
                preview += "\n..."
            content += f"\n\n[dim]{preview}[/dim]"
        
        elif step.state == AgentState.COMPLETED:
            content += "\n\n[green]Task completed successfully![/green]"
        
        return Panel(
            content,
            title=f"Agent Status",
            border_style=self._get_state_color(step.state)
        )
    
    def _get_state_emoji(self, state: AgentState) -> str:
        """Get emoji for agent state."""
        emoji_map = {
            AgentState.THINKING: "🤔",
            AgentState.CALLING_TOOL: "🔧",
            AgentState.REFLECTING: "💭",
            AgentState.COMPLETED: "✅",
            AgentState.ERROR: "❌"
        }
        return emoji_map.get(state, "⚪")
    
    def _get_state_color(self, state: AgentState) -> str:
        """Get color for agent state."""
        color_map = {
            AgentState.THINKING: "blue",
            AgentState.CALLING_TOOL: "yellow",
            AgentState.REFLECTING: "magenta",
            AgentState.COMPLETED: "green",
            AgentState.ERROR: "red"
        }
        return color_map.get(state, "white")
    
    async def start(self):
        """Start the console (for compatibility)."""
        # This method is called by the agent but we handle display updates
        # through update_status, so this can be a no-op
        pass