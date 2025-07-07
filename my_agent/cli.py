import click

from my_agent.utils.cli_console import CLIConsole


@click.group()
def main():
    pass


@main.command()
@click.argument("task_string")
@click.option(
    "--config-file",
    help="Path to the configuration file. Defaults to my_agent_config.json in the current directory.",
)
@click.option(
    "--trajectory-file",
)
def run(task_string, config_file, trajectory_file):
    """Run the agent with a given task."""
    CLIConsole.run(task_string, config_file, trajectory_file)


@main.command()
@click.option(
    "--config-file",
    help="Path to the configuration file. Defaults to my_agent_config.json in the current directory.",
)
@click.option(
    "--trajectory-file",
)
def interactive(config_file, trajectory_file):
    """Run the agent in interactive mode."""
    CLIConsole.interactive(config_file, trajectory_file)


if __name__ == "__main__":
    main() 