#!/usr/bin/env python3

import asyncio
from pathlib import Path
from my_agent.tools.edit_tool import TextEditorTool
from my_agent.tools.base import ToolCallArguments

async def test_tool():
    tool = TextEditorTool()
    
    # Test creating a file
    result = await tool.execute(ToolCallArguments({
        "command": "create",
        "path": "C:\\Users\\mhossen\\OneDrive - University of Tennessee\\AI\\my-agent\\my-agent\\direct_test.txt",
        "file_text": "This file was created directly by the tool!"
    }))
    
    print(f"Tool execution result:")
    print(f"Error code: {result.error_code}")
    print(f"Output: {result.output}")
    print(f"Error: {result.error}")
    
    # Check if file exists
    file_path = Path("C:\\Users\\mhossen\\OneDrive - University of Tennessee\\AI\\my-agent\\my-agent\\direct_test.txt")
    print(f"\nFile exists: {file_path.exists()}")
    if file_path.exists():
        print(f"File content: {file_path.read_text()}")

if __name__ == "__main__":
    asyncio.run(test_tool())