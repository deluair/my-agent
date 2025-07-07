import unittest
import sys
import os
import asyncio
from unittest.mock import patch

from my_agent.tools.bash_tool import BashTool
from my_agent.tools.base import ToolCallArguments

class TestBashTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = BashTool()

    async def asyncTearDown(self):
        # Cleanup any active session
        if self.tool._session:
            self.tool._session.stop()

    async def test_tool_initialization(self):
        assert self.tool.get_name() == "bash"
        assert "Run commands in a bash shell" in self.tool.get_description()
        
        params = self.tool.get_parameters()
        param_names = [p.name for p in params]
        assert "command" in param_names
        assert "restart" in param_names

    async def test_command_error_handling(self):
        result = await self.tool.execute(ToolCallArguments({
            "command": "invalid_command_123"
        }))
        
        assert result.error_code != 0
        # 修复断言：检查错误信息是否包含'not found'或'not recognized'（Windows系统）
        assert any(s in result.error.lower() for s in ["not found", "not recognized"])

    async def test_session_restart(self):
        # 确保会话已初始化
        await self.tool.execute(ToolCallArguments({
            "command": "echo first session"
        }))
        
        # 修复：检查会话对象是否存在
        assert self.tool._session is not None
        
        # Restart and test new session
        restart_result = await self.tool.execute(ToolCallArguments({
            "restart": True
        }))
        assert "restarted" in restart_result.output.lower()
        
        # 修复：确保新会话已创建
        assert self.tool._session is not None
        
        # Verify new session works
        result = await self.tool.execute(ToolCallArguments({
            "command": "echo new session"
        }))
        assert "new session" in result.output

    async def test_successful_command_execution(self):
        result = await self.tool.execute(ToolCallArguments({
            "command": "echo hello world"
        }))
        
        # 修复：检查返回码是否为0
        assert result.error_code == 0
        assert "hello world" in result.output
        assert result.error == ""

    async def test_missing_command_handling(self):
        result = await self.tool.execute(ToolCallArguments({}))
        assert "no command provided" in result.error.lower()
        assert result.error_code == -1

    async def test_all(self):
        await self.asyncTearDown()
        await self.test_tool_initialization()
        await self.test_command_error_handling()
        await self.test_session_restart()
        await self.test_successful_command_execution()
        await self.test_missing_command_handling()
