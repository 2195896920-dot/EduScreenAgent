# assess_agent/core/node.py
import asyncio
import time
from typing import Any, Dict, Optional, Tuple


class EdgeBuilder:
    """解决 '-' 和 '>>' 运算符优先级问题：
    node - "action" >> next_node
    先执行 '-' 返回 EdgeBuilder，再执行 '>>' 注册到 edges"""
    
    def __init__(self, node: "Node", action: str):
        self.node = node
        self.action = action

    def __rshift__(self, target: "Node") -> "Node":
        self.node.edges[self.action] = target
        return target


class Node:
    """节点基类：做一件事，返回 (action, payload)，由 Flow 决定下一步去哪"""

    def __init__(self, max_retries: int = 1, wait: float = 0):
        self.edges: Dict[str, "Node"] = {}  # 路由表：action -> next_node
        self.max_retries = max_retries      # 失败重试次数
        self.wait = wait                  # 重试间隔（秒）

    # 语法糖：node - "action" >> next_node
    def __sub__(self, action: str) -> EdgeBuilder:
        if not isinstance(action, str):
            raise TypeError("Action must be a string")
        return EdgeBuilder(self, action)

    # 语法糖：node >> next_node（默认分支）
    def __rshift__(self, other: "Node") -> "Node":
        self.edges["default"] = other
        return other

    def route(self, action: str, payload: Any, ctx: Dict[str, Any]) -> Tuple[Optional["Node"], Any]:
        """根据 action 找下一个节点，找不到走 default"""
        if action in self.edges:
            return self.edges[action], payload
        elif "default" in self.edges:
            return self.edges["default"], payload
        return None, payload

    async def exec(self, payload: Any, ctx: Dict[str, Any]) -> Tuple[str, Any]:
        """子类必须覆盖：接收输入数据和上下文，返回 (action, output)"""
        raise NotImplementedError

    async def _exec(self, payload: Any, ctx: Dict[str, Any]) -> Tuple[str, Any]:
        """带重试保护的 exec"""
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return await self.exec(payload, ctx)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1 and self.wait > 0:
                    await asyncio.sleep(self.wait)
        # 重试用尽，抛出最后一次异常
        raise last_error if last_error else RuntimeError("Unexpected error in Node._exec")


class Flow:
    """流程引擎：按 edges 路由表驱动节点执行"""

    def __init__(self, start_node: Optional[Node] = None):
        self.start_node = start_node

    async def run(self, payload: Any, ctx: Dict[str, Any]) -> Tuple[str, Any]:
        """从 start_node 出发，沿着 edges 一路执行，直到没有下一个节点"""
        current = self.start_node
        while current:
            action, payload = await current._exec(payload, ctx)
            next_node, payload = current.route(action, payload, ctx)
            current = next_node
        return action, payload