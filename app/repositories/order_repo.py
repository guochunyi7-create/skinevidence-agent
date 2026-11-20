"""订单仓储层（示例用内存数据，接入真实 OMS 时只替换本文件）。"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import NotFoundError


@dataclass
class OrderInfo:
    order_no: str
    status: str
    carrier: str
    tracking_no: str
    items: str
    shipped_at: str

    def to_sentence(self) -> str:
        return (
            f"订单 {self.order_no} 当前状态：{self.status}；"
            f"商品：{self.items}；承运商：{self.carrier}，运单号 {self.tracking_no}，{self.shipped_at} 发出。"
        )


class OrderRepo:
    _DEMO: dict[str, OrderInfo] = {
        "202611080001": OrderInfo(
            "202611080001", "已发货", "顺丰速运", "SF1234567890",
            "水感保湿精华 30ml × 1", "2026-11-09 14:20",
        ),
        "202611100002": OrderInfo(
            "202611100002", "待发货", "待分配", "-",
            "氨基酸洁面乳 120g × 2", "-",
        ),
        "202611050003": OrderInfo(
            "202611050003", "已签收", "中通快递", "ZT9876543210",
            "清透防晒乳 SPF50+ 50ml × 1", "2026-11-05 09:10",
        ),
    }

    async def get(self, order_no: str) -> OrderInfo:
        info = self._DEMO.get(order_no.strip())
        if info is None:
            raise NotFoundError(f"未查询到订单 {order_no}")
        return info
