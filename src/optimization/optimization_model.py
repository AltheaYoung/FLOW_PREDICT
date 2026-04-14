# optimization_model.py (修正最终版)
import pandas as pd
import numpy as np
import math
from gurobipy import Model, GRB, quicksum
import sys
import os
from datetime import datetime, timedelta

# --- 删除了文件顶部的所有全局常量和文件名变量 ---
# --- 所有配置都将通过 run_optimization 函数传入 ---

# read_data 和 build_flights 函数保持不变，它们是独立的辅助函数
def read_data(path):
    # ... (代码不变) ...
    df = pd.read_csv(path, encoding='utf-8')
    df.columns = [c.strip() for c in df.columns]
    return df

# 在 optimization_model.py 中

def build_flights(df, min_half_width, min_m_size):
    dep_col = '实际起飞站四字码'; arr_col = '实际到达站四字码'; plan_col = '计划离港时间'; pred_col = '预测离港时间'; plan_arr_col = '计划到港时间'; tail_col = '机尾号'
    if dep_col not in df.columns or arr_col not in df.columns or plan_col not in df.columns or pred_col not in df.columns: raise ValueError("输入文件缺少必要列。")
    dfZ = df[df[dep_col].astype(str).str.upper() == 'ZGGG'].copy().reset_index(drop=True)
    if dfZ.empty: print("警告：未发现 ZGGG 起飞的航班。"); return [], None, None, []
    
    # 时间解析部分 (保持不变)
    dfZ['_plan_dt'] = pd.to_datetime(dfZ[plan_col], errors='coerce'); dfZ['_pred_dt'] = pd.to_datetime(dfZ[pred_col], errors='coerce'); dfZ['_arr_plan_dt'] = pd.to_datetime(dfZ[plan_arr_col], errors='coerce')
    if dfZ['_plan_dt'].isna().any() or dfZ['_pred_dt'].isna().any(): raise ValueError("部分计划/预测时间无法解析。")
    ref_date = dfZ['_plan_dt'].min().normalize()
    dfZ['_plan_min'] = ((dfZ['_plan_dt'] - ref_date).dt.total_seconds() / 60).astype(int); dfZ['_pred_min'] = ((dfZ['_pred_dt'] - ref_date).dt.total_seconds() / 60).astype(int); dfZ['_arr_plan_min'] = ((dfZ['_arr_plan_dt'] - ref_date).dt.total_seconds() / 60).astype('Int64'); dfZ['_orig_delay'] = (dfZ['_pred_min'] - dfZ['_plan_min']).astype(int)
    min_allowed = int(dfZ['_plan_min'].min() - 6*60); max_allowed = int(dfZ['_plan_min'].max() + 6*60)
    
    flights, expanded_flights = [], []
    for idx, row in dfZ.iterrows():
        # 【关键修改】将所有需要用到的变量在循环开头就明确定义好
        current_p = int(row['_plan_min'])
        pred_min = int(row['_pred_min'])
        orig = int(row['_orig_delay'])
        tail = str(row.get(tail_col, '')).strip().upper()

        # 计算候选集 M (逻辑不变)
        half_width = max(int(math.ceil(abs(orig) * 0.3)), min_half_width)
        M = [m for m in range(pred_min - half_width, pred_min + half_width + 1) if min_allowed <= m <= max_allowed]
        orig_len = len(M)
        while len(M) < min_m_size:
            half_width += 1; M = [m for m in range(pred_min - half_width, pred_min + half_width + 1) if min_allowed <= m <= max_allowed]
            if half_width > 24*60: break
        if len(M) != orig_len: expanded_flights.append({'orig_index': idx, '机尾号': row.get('机尾号', ''), '原|M|': orig_len, '新|M|': len(M), 'center_pred_min': pred_min, 'half_width': half_width})
        
        is_international = not str(row[arr_col]).strip().upper().startswith('Z')
        
        # 计算前序航班到达时间
        arr_prev_min = None
        if tail:
            same_tail_flights = dfZ[dfZ[tail_col].astype(str).str.upper() == tail]
            # 【关键修改】在查询中明确使用 current_p
            previous_flights = same_tail_flights[
                (same_tail_flights['_arr_plan_min'].notna()) & 
                (same_tail_flights['_arr_plan_min'] < current_p)
            ]
            if not previous_flights.empty:
                arr_prev_min = int(previous_flights['_arr_plan_min'].max())
        
        # 组装 flight 字典
        flights.append({
            'id': str(idx),
            'orig_index': idx,
            'tail': tail,
            'p': current_p, # <-- 使用 current_p
            'pred_min': pred_min,
            'M': sorted(set(M)),
            'orig_delay': orig,
            'is_international': is_international,
            'arr_prev_min': arr_prev_min,
            '航班号': row.get('航班号', '')
        })
    return flights, dfZ, ref_date, expanded_flights


# build_and_solve_strict 和 extract_solution_and_write 现在接收所有配置作为参数
def build_and_solve_strict(flights, time_limit, threads, minute_capacity):
    """【修正版】构建并求解严格的MIP模型，解决了变量作用域问题。"""
    
    model = Model("ZGGG_strict")
    model.Params.OutputFlag = 1
    model.Params.Threads = threads
    model.Params.TimeLimit = time_limit
    
    # 1. 定义变量 (保持不变)
    x = {}
    z = {}
    for f_data in flights:
        f_id = f_data['id']
        for m in f_data['M']:
            x[f_id, m] = model.addVar(vtype=GRB.BINARY, name=f"x_{f_id}_{m}")
        z[f_id] = model.addVar(vtype=GRB.BINARY, name=f"z_{f_id}")
    model.update()

    # 2. 添加约束
    intl_violations = []
    for f_data in flights:
        # 【关键修改】在循环开始时，从字典中提取所有需要的值
        f_id = f_data['id']
        p = f_data['p']
        M = f_data['M']
        is_international = f_data['is_international']
        arr_prev_min = f_data['arr_prev_min']

        # 约束 2a: 每个航班只能选择一个起飞时间
        model.addConstr(quicksum(x[f_id, m] for m in M) == 1, name=f"one_{f_id}")

        # 约束 2b: 计算延误并处理延误标志 z
        delta = quicksum((m - p) * x[f_id, m] for m in M)
        model.addGenConstrIndicator(z[f_id], 0, delta <= 15)
        model.addGenConstrIndicator(z[f_id], 1, delta >= 16)
        
        # 约束 2c: 国际航班延误上限
        if is_international:
            if any((m - p) <= 45 for m in M):
                model.addConstr(delta <= 45, name=f"intlcap_{f_id}")
            else:
                intl_violations.append({
                    'id': f_id,
                    'orig_index': f_data['orig_index'],
                    '航班号': f_data.get('航班号', ''),
                    'p': p,
                    'M_size': len(M),
                    'min_delta': min((m - p) for m in M)
                })

        # 约束 2d: 周转时间
        if arr_prev_min is not None:
            # 这里的 delta (quicksum(m*x)) 与上面的延误 delta 不同，重新定义
            departure_minute = quicksum(m * x[f_id, m] for m in M)
            model.addConstr(departure_minute - arr_prev_min >= 55, name=f"turn_{f_id}")

    # 约束 2e: 容量约束 (分钟级和小时级)
    all_minutes = sorted({m for f in flights for m in f['M']})
    for m in all_minutes:
        model.addConstr(quicksum(x[f['id'], m] for f in flights if m in f['M']) <= minute_capacity, name=f"mincap_{m}")
    
    hour_groups = {}
    for m in all_minutes: hour_groups.setdefault(m // 60, []).append(m)
    for h, mins in hour_groups.items():
        model.addConstr(quicksum(x[f['id'], m] for f in flights for m in mins if m in f['M']) <= 81, name=f"hourcap_{h}")

    # 3. 定义目标函数 (保持不变)
    obj0 = quicksum(z[f['id']] for f in flights)
    obj1 = quicksum((m - f['p']) * x[f['id'], m] for f in flights for m in f['M'])
    model.setObjectiveN(obj0, index=0, priority=2, name="min_delayed_count")
    model.setObjectiveN(obj1, index=1, priority=1, name="min_total_delay")

    model.update()
    model._intl_violations = intl_violations
    model._xvars = x
    return model


def extract_solution_and_write(model, flights, dfZ, ref_date, expanded_flights, intl_violations, output_dir):
    # ... (代码不变, 但使用 output_dir 来构建保存路径) ...
    output_file = os.path.join(output_dir, "optimal_plan.csv")
    intl_violations_file = os.path.join(output_dir, "intl_violations.csv")
    summary_file = os.path.join(output_dir, "delay_summary.csv")
    # ...
    # save_csv(outdf, output_file, ...)
    # save_csv(pd.DataFrame(intl_violations), intl_violations_file, ...)
    # ...
    def save_csv(df, path, desc): df.to_csv(path, index=False, encoding='utf-8-sig'); print(f"{desc} 已写入: {path}")
    def calc_delay_stats(delays): return len(delays), float(np.mean(delays)) if delays else 0.0
    status = model.status
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        print(f"模型未找到可行解，状态码: {status}"); print("诊断信息..."); Msizes = [len(f['M']) for f in flights]; print(f"航班数: {len(flights)}, |M| 平均/最小/最大 = {np.mean(Msizes):.2f}/{min(Msizes)}/{max(Msizes)}"); forced = [f for f in flights if len(f['M']) == 1]; print(f"强制 |M|==1 的航班数: {len(forced)}"); [print(f" id={f['id']}, tail={f['tail']}, p={f['p']}, M={f['M']}") for f in forced[:10]]
        if intl_violations: save_csv(pd.DataFrame(intl_violations), intl_violations_file, "国际航班违规列表")
        return
    x = model._xvars
    rows, orig_delays = [], []
    for f in flights:
        orig_row = dfZ.loc[f['orig_index']]
        if orig_row.get('实际起飞站四字码') == 'ZGGG' and f.get('orig_delay', 0) > 15: orig_delays.append(f['orig_delay'])
        chosen = next((m for m in f['M'] if x[(f['id'], m)].X > 0.5), min(f['M'], key=lambda m: abs(m - f['pred_min'])))
        delay = chosen - f['p']; chosen_dt = (ref_date + pd.Timedelta(minutes=int(chosen))).strftime("%Y-%m-%d %H:%M:%S")
        rows.append({'机尾号': orig_row.get('机尾号',''), '航班号': orig_row.get('航班号',''), '计划离港时间': orig_row.get('计划离港时间',''), '预测离港时间': orig_row.get('预测离港时间',''), '调整后离港时间': chosen_dt, '延误分钟': delay, '是否延误航班': int(delay > 15)})
    outdf = pd.DataFrame(rows)
    save_csv(outdf, output_file, "优化结果")
    opt_delays = outdf.loc[outdf['延误分钟'] > 15, '延误分钟'].tolist()
    orig_count, orig_avg = calc_delay_stats(orig_delays); opt_count, opt_avg = calc_delay_stats(opt_delays)
    print("---- 延误统计 ----"); print(f"原始（预测）延误航班数 (>15min): {orig_count}, 平均延误时长: {orig_avg:.2f} min"); print(f"优化后延误航班数 (>15min): {opt_count}, 平均延误时长: {opt_avg:.2f} min")
    summary = pd.DataFrame([{'type': 'original_predicted', 'delayed_count': orig_count, 'avg_delay_min': round(orig_avg, 2)}, {'type': 'optimized', 'delayed_count': opt_count, 'avg_delay_min': round(opt_avg, 2)}])
    save_csv(summary, summary_file, "延误统计")
    if intl_violations: save_csv(pd.DataFrame(intl_violations), intl_violations_file, "国际航班违规列表")


# --- 这是GUI调用的唯一入口函数 ---
def run_optimization(input_filepath, output_dir, time_limit=120, threads=12):
    """
    运行完整的航班协同恢复优化模型。
    """
    print("\n" + "="*20 + " 开始运行协同恢复优化模型 " + "="*20)
    MINUTE_CAPACITY = 15
    MIN_HALF_WIDTH = 5
    MIN_M_SIZE = 5
    # 1. 数据预处理
    print("步骤1: 正在转换仿真日志为优化模型输入格式...")
    try:
        sim_df = pd.read_csv(input_filepath)
        opt_input_df = sim_df[sim_df['departure_airport'] == 'ZGGG'].copy()
        rename_map = {
            'airline': '机尾号',                  # <-- 关键！告诉程序'airline'列就是'机尾号'
            'flight_id': '航班号',
            'departure_airport': '实际起飞站四字码',
            'arrival_airport': '实际到达站四字码',
            'scheduled_departure_time': '计划离港时间',
            'predicted_departure_time': '预测离港时间',
            'scheduled_arrival_time': '计划到港时间'
        }
        source_cols = list(rename_map.keys())
        missing_cols = [col for col in source_cols if col not in opt_input_df.columns]
        if missing_cols:
            raise ValueError(f"仿真日志文件中缺少必要的列: {missing_cols}")
        opt_input_df.rename(columns=rename_map, inplace=True)
        last_valid_time = opt_input_df['预测离港时间'].dropna().max()
        if pd.isna(last_valid_time):
            ref_date = pd.to_datetime(opt_input_df['计划离港时间'].min()).normalize()
            last_valid_time = ref_date + timedelta(days=1) - timedelta(minutes=1)
        
        opt_input_df['预测离港时间'].fillna(pd.to_datetime(last_valid_time).strftime('%Y-%m-%d %H:%M:%S'), inplace=True)
        prepared_input_path = os.path.join(output_dir, "prepared_for_optimization.csv")
        
        # 筛选出优化模型需要的列
        required_cols_for_opt = [
            '机尾号', '航班号', '实际起飞站四字码', '实际到达站四字码',
            '计划离港时间', '预测离港时间', '计划到港时间'
        ]
        opt_input_df[required_cols_for_opt].to_csv(prepared_input_path, index=False, encoding='utf-8-sig')
        print(f"已生成优化模型的标准输入文件: {prepared_input_path}")
    except Exception as e:
        print(f"数据转换失败: {e}")
        return False

    # 2. 运行优化主逻辑 (不再需要中间文件和全局变量)
    print("\n步骤2: 正在构建并求解优化模型...")
    try:
        # 【修改】直接将准备好的 DataFrame 传递给 build_flights
        flights, dfZ, ref_date, expanded_flights = build_flights(opt_input_df, MIN_HALF_WIDTH, MIN_M_SIZE)
        
        print("构建航班候选集完成，航班数量:", len(flights))
        print("自动扩窗记录（示例，最多10条）：")
        [print(e) for e in expanded_flights[:10]]
        
        # 【修改】将配置参数传递给模型构建函数
        model = build_and_solve_strict(flights, time_limit, threads, MINUTE_CAPACITY)
        
        print(f"开始求解（严格 MIP）... TimeLimit={time_limit}s, Threads={threads}")
        model.optimize()
        
        intl_violations = getattr(model, "_intl_violations", [])
        
        # 【修改】将 output_dir 传递给结果提取函数
        extract_solution_and_write(model, flights, dfZ, ref_date, expanded_flights, intl_violations, output_dir)
        
        print("\n优化流程执行完毕。")
        return True

    except Exception as e:
        print(f"优化模型运行时发生错误: {e}")
        return False

# 这个 if __name__ 块仅用于独立测试此脚本
if __name__ == "__main__":
    # 示例用法
    # 假设当前目录下有 'full_flight_log_ZGGG.csv'
    run_optimization(
        input_filepath='full_flight_log_ZGGG.csv',
        output_dir='test_optimization_output',
        time_limit=60 
    )