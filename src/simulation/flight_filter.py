import pandas as pd

def filter_departure_flights(
    input_filepath: str, 
    output_filepath: str, 
    target_airport_code: str,
    target_date_str: str # 新增：目标日期字符串
) -> None:
    """
    从一个大的航班数据文件中，筛选出特定机场的出港航班，并保存为新文件。

    Args:
        input_filepath (str): 原始的、完整的航班数据文件路径。
        output_filepath (str): 清理后要保存的文件路径。
        target_airport_code (str): 目标出港机场的四字码，例如 "ZGGG"。
    """
    print(f"开始处理文件: {input_filepath}")
    
    try:
        # 使用和main.py中相同的参数来读取文件，确保兼容性
        # sep='\t' 表示文件是Tab分隔的
        # encoding='utf-8-sig' 处理带BOM的UTF-8文件
        df = pd.read_excel(input_filepath, dtype=str).fillna('')
        print(f"成功加载 {len(df)} 条总航班记录。")
        
    except FileNotFoundError:
        print(f"错误: 输入文件 {input_filepath} 未找到。请检查文件名和路径。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # --- 核心筛选逻辑 ---
    # 定义出港机场的列名
    departure_column = '计划起飞站四字码'
    time_column = '计划离港时间' # 我们按计划离港时间来筛选日期
    
    if departure_column not in df.columns:
        print(f"错误: 文件中未找到关键列 '{departure_column}'。请检查CSV/TSV文件的表头。")
        return
        
    # 执行筛选：只保留 '计划起飞站四字码' 等于目标机场的行
    # 1. 筛选出港机场
    df_dep = df[df[departure_column] == target_airport_code].copy()
    
    # 2. 筛选日期
    #    将时间列转换为datetime对象，以便按日期比较
    df_dep['parsed_time'] = pd.to_datetime(df_dep[time_column], errors='coerce')
    #    只保留日期部分与目标日期一致的行
    filtered_df = df_dep[df_dep['parsed_time'].dt.date == pd.to_datetime(target_date_str).date()].copy()
    
    print(f"筛选完成: 找到 {len(filtered_df)} 条从 {target_airport_code} 在日期 {target_date_str} 出港的航班。")
    # --- 准备并保存输出文件 ---
    # 定义我们的仿真程序需要的列
    required_columns = [
        '机尾号',
        '航班号',
        '计划起飞站四字码',
        '计划到达站四字码',
        '计划离港时间',
        '计划到港时间',
        '实际离港时间',
        '实际到港时间'

    ]
    
    # 确保所有需要的列都存在于筛选后的数据中
    output_df = filtered_df[required_columns]
    
    try:
        # 将筛选结果保存为新的Tab分隔文件，不包含索引列
        output_df.to_csv(output_filepath, sep='\t', index=False, encoding='utf-8-sig')
        print(f"\n[+] 成功! 已将筛选结果保存到: {output_filepath}")
        print("您现在可以在 main.py 中使用这个新文件作为输入。")
        
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 请在这里配置您的文件和机场 ---
    
    # 1. 输入文件名：您提供的包含所有航班的大文件的名字
    #    请将 'your_full_flights_data.csv' 替换为您的实际文件名
    full_data_filepath = "data.xlsx"
    # 3. 目标机场：广州白云国际机场
    TARGET_AIRPORT = 'ZGGG'
    TARGET_DATE = '2025-05-23'
    # 2. 输出文件名：我们想要生成的新文件的名字
    zggg_departures_filepath = f'zggg_departures_only_{TARGET_DATE}.csv'
    

    
    # --- 运行筛选函数 ---
    filter_departure_flights(full_data_filepath, zggg_departures_filepath, TARGET_AIRPORT,TARGET_DATE)