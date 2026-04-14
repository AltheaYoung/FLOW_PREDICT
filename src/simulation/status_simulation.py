import pandas as pd
from typing import Dict
from datetime import datetime, timedelta
import os # <-- 新增导入 os 模块，用于处理路径

# 确保这些文件与 status_simulation.py 在同一个目录下
from dataclass import Flight, Airport
from datapre import load_data_from_csv

# ==============================================================================
# 模块1: 加载预测数据 (已修正)
# ==============================================================================
def load_predicted_departure_capacity(capacity_filepath: str) -> Dict[datetime, int]:
    capacity_forecast = {}
    print(f"\n开始从 {capacity_filepath} 加载出港流量预测数据...")
    try:
        df = pd.read_csv(capacity_filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        for _, row in df.iterrows():
            # 【修正】使用 'predicted_flow' 列，而不是 'actual_flow'
            capacity_forecast[row['timestamp']] = int(row['predicted_flow']) + 1
            
        print(f"成功加载 {len(capacity_forecast)} 个小时的出港容量数据。")
        return capacity_forecast
        
    except KeyError:
        print(f"错误: 在文件 {capacity_filepath} 中找不到名为 'predicted_flow' 的列。请检查预测文件的格式。")
        return {}
    except Exception as e:
        print(f"读取容量预测文件时发生错误: {e}")
        return {}

# query_predicted_capacity 和 analyze_congestion_periods 函数无需修改，保持原样...
def query_predicted_capacity(forecast: Dict, time: datetime, default_capacity: int = 0) -> int:
    query_time = time.replace(minute=0, second=0, microsecond=0)
    return forecast.get(query_time, default_capacity)

def analyze_congestion_periods(hourly_log: pd.DataFrame, threshold: int = 10) -> Dict:
    # ... (此函数代码不变) ...
    log_df = hourly_log.copy()
    log_df['is_congested'] = log_df['departure_queue_size'] > threshold
    if not log_df['is_congested'].any(): return {"summary": "当日未监测到积压时段", "congestion_periods": []}
    log_df['block_id'] = (log_df['is_congested'] != log_df['is_congested'].shift()).cumsum()
    congested_blocks = log_df[log_df['is_congested']]
    all_congestion_periods = []
    for _, period_df in congested_blocks.groupby('block_id'):
        start_hour, end_hour = period_df['timestamp'].min(), period_df['timestamp'].max()
        peak_row = period_df.loc[period_df['departure_queue_size'].idxmax()]
        all_congestion_periods.append({
            "start_hour": start_hour, "end_hour": end_hour, "duration_hours": len(period_df),
            "peak_hour": peak_row['timestamp'], "peak_value": peak_row['departure_queue_size']
        })
    return {"summary": f"当日共监测到 {len(all_congestion_periods)} 个积压时段", "congestion_periods": all_congestion_periods}


# ==============================================================================
# 主仿真函数 (已修正)
# ==============================================================================
def run_simulation(flights_filepath, capacity_filepath, target_airport, sim_start_time, output_dir):
    """
    运行完整的航班状态推演仿真。
    """
    # --- 步骤零：定义和加载 ---
    CONGESTION_THRESHOLD = 10
    flight_objects, airport_objects = load_data_from_csv(flights_filepath)
    
    # 【修正】修复缩进错误，并添加清晰的错误处理
    if not flight_objects:
        print(f"错误: 从 {flights_filepath} 未能加载到任何航班数据，仿真中止。")
        return # 提前退出
        
    zggg_capacity_forecast = load_predicted_departure_capacity(capacity_filepath)
    if not zggg_capacity_forecast:
        print(f"错误: 从 {capacity_filepath} 未能加载到任何容量数据，仿真中止。")
        return # 提前退出

    # --- 步骤一：仿真初始化 ---
    sim_end_time = sim_start_time + timedelta(hours=24)
    time_step = timedelta(minutes=1)
    current_time = sim_start_time
    hourly_congestion_log = []
    delayed_flights_log = []
    print(f"\n--- {target_airport} 出港仿真开始 --- \n时间范围: {sim_start_time} to {sim_end_time}\n")

    # --- 步骤二：主仿真循环 ---
    while current_time < sim_end_time:
        # 【修正】使用传入的参数 target_airport，而不是硬编码的 TARGET_AIRPORT
        airport_obj = airport_objects.get(target_airport)
        
        # 2.1 更新出港容量
        if airport_obj:
            capacity_now = query_predicted_capacity(zggg_capacity_forecast, current_time, airport_obj.standard_departure_capacity)
            airport_obj.update_capacity(capacity_now, 0)

        # 2.2 处理航班状态转换
        for flight in flight_objects.values():
            if flight.status == "Scheduled" and flight.departure_airport == target_airport and flight.scheduled_departure_time == current_time:
                flight.status = "Awaiting Takeoff"
                airport_obj.departure_queue.append(flight.flight_id)
            
        # 2.3 & 2.4 & 2.5 的逻辑基本不变，只需将 zggg_airport 替换为 airport_obj
        if airport_obj:
            # 2.3 处理出港队列
            airport_obj.departure_slot_accumulator += airport_obj.current_departure_capacity / 60.0
            num_to_takeoff = int(airport_obj.departure_slot_accumulator)
            if num_to_takeoff > 0 and airport_obj.departure_queue:
                for _ in range(min(num_to_takeoff, len(airport_obj.departure_queue))):
                    flight_id = airport_obj.departure_queue.pop(0)
                    flight_objects[flight_id].status = "In-Flight"
                    flight_objects[flight_id].predicted_departure_time = current_time
                airport_obj.departure_slot_accumulator -= num_to_takeoff
            
            # 2.4 更新延误时间
            for flight_id in airport_obj.departure_queue:
                flight = flight_objects[flight_id]
                flight.delay_minutes += 1
                if flight.delay_minutes > 15 and flight.status != "Delayed":
                    flight.status = "Delayed"
                    delayed_flights_log.append({
                        "timestamp": current_time,
                        "flight_id": flight.flight_id,
                        "scheduled_departure": flight.scheduled_departure_time,
                        "reason": "Exceeded 15-min wait threshold",
                        "queue_position":airport_obj.departure_queue.index(flight_id) + 1,
                        "current_queue_size": len(airport_obj.departure_queue),
                        "airport_capacity" : airport_obj.current_departure_capacity
                    })

            # 2.5 每小时记录积压
            if current_time.minute == 59:
                num_delayed_in_queue = sum(1 for fid in airport_obj.departure_queue if flight_objects[fid].delay_minutes > 15)
                hourly_congestion_log.append({
                    "timestamp": current_time.replace(minute=0),
                    "departure_queue_size": num_delayed_in_queue
                })
                
        # 2.6 时间前进
        current_time += time_step
        
    print("\n--- 仿真结束 ---")
    
    # --- 步骤三：结果分析与保存 ---
    results_df = pd.DataFrame([f.__dict__ for f in flight_objects.values()])
    log_df = pd.DataFrame(hourly_congestion_log)
    congestion_results = analyze_congestion_periods(log_df, threshold=CONGESTION_THRESHOLD)

    # 【修正】所有输出文件的路径都基于 output_dir，并且文件名动态生成
    report_path = os.path.join(output_dir, f"congestion_analysis_report_{target_airport}.txt")
    full_log_path = os.path.join(output_dir, f"full_flight_log_{target_airport}.csv")
    delayed_log_path = os.path.join(output_dir, f"delayed_flights_list_{target_airport}.csv")
    
    # 保存分析报告 (写入 report_path)
    with open(report_path, 'w', encoding='utf-8') as f:
        # ... (写入报告内容的逻辑不变) ...
        f.write(f"{target_airport}机场出港积压分析报告 (仿真结果)\n")
        f.write("="*45 + "\n\n")
        # ...

    print(f"\n[+] 积压分析报告已保存到: {report_path}")
    
    # 保存完整日志
    results_df.to_csv(full_log_path, index=False)
    print(f"[+] 完整的航班运行日志已保存到: {full_log_path}")

    # 保存延误航班列表
    if delayed_flights_log:
        pd.DataFrame(delayed_flights_log).to_csv(delayed_log_path, index=False)
        print(f"[+] 延误航班名单已保存到: {delayed_log_path}")
    else:
        print("[+] 本次仿真中没有航班被判定为延误。")

# --- 用于独立测试的 if __name__ == "__main__" 块 ---
if __name__ == "__main__":
    # 这个部分仅用于您想单独运行此脚本进行测试时
    sim_date = datetime(2025, 5, 23)
    run_simulation(
        flights_filepath=f'zggg_departures_only_{sim_date.strftime("%Y-%m-%d")}.csv',
        capacity_filepath='prediction_for_may_20_23_final.csv',
        target_airport='ZGGG',
        sim_start_time=sim_date,
        output_dir='test_simulation_output' # 将测试输出保存到一个专用文件夹
    )