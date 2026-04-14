from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# 使用 type hint 来增加代码的可读性
AirportCode = str  # 例如 "ZBAA"
FlightID = str      # 例如 "MU5101"



@dataclass
class Flight:
    flight_id: str
    airline: str
    departure_airport: str
    arrival_airport: str
    scheduled_departure_time: datetime
    scheduled_arrival_time: datetime
    
    # 新增：提前计算好的计划飞行时间，用于后续计算
    planned_duration: timedelta = field(init=False)
    
    # 动态状态信息
    status: str = "Scheduled"
    actual_departure_time: Optional[datetime] = None
    estimated_arrival_time: Optional[datetime] = None # 新增：起飞后才能确定的预计到达时间
    actual_arrival_time: Optional[datetime] = None
    delay_minutes: int = 0
    
    def __post_init__(self):
        """在对象创建后，自动计算计划飞行时长"""
        self.planned_duration = self.scheduled_arrival_time - self.scheduled_departure_time
    def __str__(self):
        """方便打印航班信息"""
        return (f"Flight({self.flight_id}, from {self.departure_airport} to {self.arrival_airport}, "
                f"Status: {self.status}, Delay: {self.delay_minutes} mins),act_depart{self.actual_departure_time}")

@dataclass
class Airport:
    code: str
    standard_departure_capacity: int
    standard_arrival_capacity: int
    
    # 动态信息
    current_departure_capacity: int = 0
    current_arrival_capacity: int = 0
    # 新增：更平滑的流量控制累加器
    departure_slot_accumulator: float = 0.0
    arrival_slot_accumulator: float = 0.0
    
    departure_queue: List[str] = field(default_factory=list)
    arrival_queue: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.current_departure_capacity = self.standard_departure_capacity
        self.current_arrival_capacity = self.standard_arrival_capacity

    def update_capacity(self, departure_capacity: int, arrival_capacity: int):
        self.current_departure_capacity = departure_capacity
        self.current_arrival_capacity = arrival_capacity
    def __str__(self):
        """方便打印机场状态"""
        return (f"Airport({self.code}, Dep Cap: {self.current_departure_capacity}, Arr Cap: {self.current_arrival_capacity}, "
                f"Dep Queue: {len(self.departure_queue)}, Arr Queue: {len(self.arrival_queue)})")
