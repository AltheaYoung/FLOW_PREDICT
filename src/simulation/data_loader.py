import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from dataclass import Flight ,Airport

# --- 修正后的数据加载与实例化函数 ---

def load_data_from_csv(
    flights_filepath: str,
    default_capacity_per_hour: int = 10
) -> (Dict[str, Flight], Dict[str, Airport]):
    """
    从CSV文件加载航班数据，并创建Flight和Airport对象。
    此版本假定时间列是标准日期时间字符串。
    """
    flight_objects = {}
    airport_objects = {}

    print(f"开始从 {flights_filepath} 加载数据...")

    try:
        # 使用pandas读取CSV
        df = pd.read_csv(flights_filepath, sep='\t',dtype=str).fillna('') # 读取所有列为字符串，并将NaN填充为空字符串
        print(df)
    except FileNotFoundError:
        print(f"错误: 文件 {flights_filepath} 未找到。")
        return {}, {}
    except Exception as e:
        print(f"读取CSV时发生错误: {e}")
        return {}, {}
    
    # 根据你的截图，我们来指定更精确的列名
    # 请务必根据你的CSV文件的实际第一行（表头）来修改这里的中文！
    flight_data_columns = {
        '机尾号': 'tail_id',
        '航班号': 'flight_id',              # 假设有这一列
        '计划起飞站四字码': 'departure_airport',
        '计划到达站四字码': 'arrival_airport',
        '计划离港时间': 'scheduled_departure_time',
        '计划到港时间': 'scheduled_arrival_time',
        # '预测离港时间': 'actual_departure_time',
        # '实际到港时间': 'actual_arrival_time',
        '预测离港时间': 'predicted_departure_time'
    }
    # 为了鲁棒性，先检查列是否存在，如果不存在则重命名，避免程序崩溃
    for ch_name, en_name in flight_data_columns.items():
        if ch_name in df.columns:
            df.rename(columns={ch_name: en_name}, inplace=True)

    # 如果没有航班号列，我们创建一个
    if 'flight_id' not in df.columns:
        df['flight_id'] = [f"FL{i:04d}" for i in range(len(df))]
        print("没有航班号列")

    print(f"成功加载 {len(df)} 条航班记录。开始实例化对象...")

    for index, row in df.iterrows():
        try:
            # --- 1. 直接使用pd.to_datetime解析标准时间字符串 ---
            # `errors='coerce'` 是一个非常有用的参数：
            # 如果某个单元格无法被解析（比如它是空的或文本'--'），
            # pandas会将其转换为NaT (Not a Time)，而不是报错。
            sch_dep_time = pd.to_datetime(row['scheduled_departure_time'], errors='coerce')
            sch_arr_time = pd.to_datetime(row['scheduled_arrival_time'], errors='coerce')
            # act_dep_time = pd.to_datetime(row['actual_departure_time'], errors='coerce')
            #ct_arr_time = pd.to_datetime(row['actual_arrival_time'], errors='coerce')

            # 如果任何一个时间是无效的，就跳过这一行
            if pd.isna(sch_dep_time) or pd.isna(sch_arr_time):
                print(f"信息: 第 {index+2} 行时间数据无效，已跳过。")
                continue

            # --- 2. 获取其他信息 ---
            dep_airport_code = row['departure_airport'].strip()
            arr_airport_code = row['arrival_airport'].strip()
            flight_id = row['flight_id'].strip()
            tail_id = row['tail_id'].strip()
            
            # 如果机场代码为空，也跳过
            if not dep_airport_code or not arr_airport_code:
                print("机场代码为空")
                continue

            # --- 3. 创建 Flight 对象 ---
            new_flight = Flight(
                flight_id=flight_id,
                airline=tail_id, # 简化处理，用航班号前两位做航司代码
                departure_airport=dep_airport_code,
                arrival_airport=arr_airport_code,
                scheduled_departure_time=sch_dep_time,
                scheduled_arrival_time=sch_arr_time,
            )
            flight_objects[flight_id] = new_flight

            # --- 4. 动态创建 Airport 对象 (如果不存在) ---
            if dep_airport_code not in airport_objects:
                airport_objects[dep_airport_code] = Airport(
                    code=dep_airport_code,
                    standard_departure_capacity=default_capacity_per_hour,
                    standard_arrival_capacity=default_capacity_per_hour
                )
            if arr_airport_code not in airport_objects:
                airport_objects[arr_airport_code] = Airport(
                    code=arr_airport_code,
                    standard_departure_capacity=default_capacity_per_hour,
                    standard_arrival_capacity=default_capacity_per_hour
                )

        except Exception as e:
            # 捕获其他可能的未知错误
            print(f"警告: 处理第 {index+2} 行时出错: {e}。已跳过。")
            pass

    print(f"实例化完成: 创建了 {len(flight_objects)} 个有效航班对象和 {len(airport_objects)} 个机场对象。")
    
    return flight_objects, airport_objects
