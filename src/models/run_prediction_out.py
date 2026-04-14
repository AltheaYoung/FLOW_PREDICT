import pandas as pd
import numpy as np
import os
import joblib
import gzip
from datetime import datetime, timedelta
from keras.models import load_model

# ==============================================================================
#  HELPER FUNCTIONS (我们之前讨论过的所有辅助函数)
# ==============================================================================

# 修改这个函数
def get_historical_flow_fallback(target_date, fallback_flow_path, flow_cols=['出港流量']):
    """
    修改后的版本：加载小时平均流量，并附加一个完整的DatetimeIndex。
    
    :param target_date: 需要构建完整时间戳的目标日期 (datetime.date object)。
    :param fallback_flow_path: 备用流量文件的路径。
    :param flow_cols: 需要的流量列。
    :return: 一个以 DatetimeIndex 为索引的、包含24行流量数据的DataFrame。
    """
    print("正在加载平均流量作为历史输入...")
    try:
        avg_flow_df = pd.read_csv(fallback_flow_path, index_col='hour')
        
        # 1. 创建标准的24小时 DatetimeIndex
        start_dt = datetime.combine(target_date, datetime.min.time())
        final_index = pd.date_range(start=start_dt, periods=24, freq='H')
        
        # 2. 使用这个标准索引创建最终的DataFrame
        historical_flow_df = pd.DataFrame(index=final_index)
        
        # 3. 获取新DataFrame中每个时间戳对应的小时 (0-23)
        hours_to_map = historical_flow_df.index.hour
        
        # 4. 使用这些小时作为键，去平均流量表中查找值并填充
        for col in flow_cols:
            if col in avg_flow_df.columns:
                historical_flow_df[col] = hours_to_map.map(avg_flow_df[col])
            else:
                historical_flow_df[col] = 0 # 如果缺少，用0填充

        historical_flow_df.fillna(0, inplace=True)
        return historical_flow_df[flow_cols]
        
    except Exception as e:
        print(f"加载平均流量失败: {e}")
        return None

def get_historical_weather(target_date, real_weather_dir, fallback_avg_path, weather_cols):
    """
    获取指定日期的历史天气数据，确保无论何种情况都返回以 DatetimeIndex 为索引的 DataFrame。
    这是一个【彻底完善】的版本。

    :param target_date: 需要获取天气的日期 (datetime.date object)。
    :param real_weather_dir: 存放所有已解析天气数据的目录。
    :param fallback_avg_path: 包含每小时平均天气数据的文件路径。
    :param weather_cols: 模型需要的天气特征列名列表。
    :return: 一个以 DatetimeIndex 为索引的、包含24行天气数据的DataFrame。
    """
    
    # 定义目标日期当天的开始时间，这在两个方案中都会用到
    start_dt = datetime.combine(target_date, datetime.min.time())
    
    # --- 方案A: 尝试加载并处理当天的真实天气数据 ---
    try:
        # ... (方案A的代码与之前版本完全相同，因为它已经返回了正确的 DatetimeIndex) ...
        real_weather_filepath = os.path.join(real_weather_dir, 'weather_OBCC_ZGGG_parsed.csv')
        print(f"步骤A: 正在尝试从 {real_weather_filepath} 加载 {target_date.strftime('%Y-%m-%d')} 的真实天气数据...")
        
        all_weather_df = pd.read_csv(real_weather_filepath)
        
        year = target_date.year
        all_weather_df['timestamp'] = pd.to_datetime(
            str(year) + '-' + all_weather_df['report_day'] + ' ' + all_weather_df['report_time'], 
            format='%Y-%m-%d %H:%M', errors='coerce'
        )
        all_weather_df.dropna(subset=['timestamp'], inplace=True)

        end_dt = start_dt + timedelta(days=1)
        target_day_weather = all_weather_df[
            (all_weather_df['timestamp'] >= start_dt) & (all_weather_df['timestamp'] < end_dt)
        ].copy()

        if target_day_weather.empty:
            raise ValueError(f"在文件中未找到日期为 {target_date.strftime('%Y-%m-%d')} 的天气记录。")

        print(f"成功找到 {len(target_day_weather)} 条当天的原始天气记录。")
        target_day_weather.set_index('timestamp', inplace=True)
        
        # 为了与训练时的特征工程对齐，确保所有需要的列都存在
        for col in weather_cols:
            if col not in target_day_weather.columns:
                target_day_weather[col] = np.nan # 如果缺少某列，则创建并填充为NaN，之后会被ffill/bfill处理
        
        resampled_weather = target_day_weather[weather_cols].resample('H').mean().ffill().bfill()
        
        hourly_index = pd.date_range(start=start_dt, periods=24, freq='H')
        final_weather = resampled_weather.reindex(hourly_index).ffill().bfill()

        if final_weather.isnull().values.any():
             raise ValueError("处理后的真实天气数据不完整，切换到备用方案。")
        
        print(f"成功将真实天气数据整理为24小时格式。")
        return final_weather[weather_cols]

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"警告: {e}")
        print("切换到备用方案B...")

    # --- 方案B: 加载并使用小时平均天气数据 ---
    try:
        print(f"步骤B: 正在从 {fallback_avg_path} 使用小时平均天气作为替代...")
        avg_weather_df = pd.read_csv(fallback_avg_path, index_col='hour')
        
        # 1. 【关键修改】创建标准的24小时 DatetimeIndex
        final_index = pd.date_range(start=start_dt, periods=24, freq='H')
        
        # 2. 【关键修改】使用这个标准的 DatetimeIndex 创建最终的DataFrame
        historical_weather_final = pd.DataFrame(index=final_index)
        
        # 3. 获取新DataFrame中每个时间戳对应的小时 (0-23)
        hours_to_map = historical_weather_final.index.hour
        
        # 4. 使用这些小时作为键，去平均天气表中查找值并填充
        for col in weather_cols:
            if col in avg_weather_df.columns:
                historical_weather_final[col] = hours_to_map.map(avg_weather_df[col])
            else:
                historical_weather_final[col] = 0 # 如果平均文件里也没有，就用0填充
        
        historical_weather_final.fillna(0, inplace=True)
            
        print("已成功生成基于小时平均值的历史天气，并设置了完整的时间戳索引。")
        return historical_weather_final[weather_cols]
        
    except FileNotFoundError:
        print(f"致命错误: 备用天气文件也未找到: {fallback_avg_path}")
        return None
    except Exception as e:
        print(f"致命错误: 加载备用天气时出错: {e}")
        return None


def engineer_features(df):
    """一个通用的特征工程函数，用于历史和未来数据。"""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['is_peak_hours'] = ((df['hour'] >= 7) & (df['hour'] <= 22)).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 6.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 6.0)
    if 'wind_dir' in df.columns:
        df['wind_dir_rad'] = np.deg2rad(df['wind_dir'])
        df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
        df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
    return df

def prepare_encoder_input(historical_date_str, work_dir, scalers):
    """
    【修正版】准备编码器的完整输入。
    """
    print("\n--- 准备编码器输入 (过去24小时) ---")
    historical_date = datetime.strptime(historical_date_str, '%Y-%m-%d').date()
    
    # 1. 获取各部分数据 (现在它们都返回带 DatetimeIndex 的 DataFrame)
    flow_df = get_historical_flow_fallback(
        historical_date,
        os.path.join(work_dir, 'hourly_flow_statistics.csv')
    )
    
    weather_cols_raw = ['temp_c', 'wind_dir', 'wind_speed_mps', 'visibility_m', 'pressure_hpa', 'dewpoint_c', 'wind_gust_mps']
    weather_df = get_historical_weather(
        historical_date, 
        work_dir, # 假设 'weather_OBCC_ZGGG_parsed.csv' 就在 work_dir
        os.path.join(work_dir, 'may_hourly_average_weather.csv'),
        weather_cols_raw
    )
    
    if flow_df is None or weather_df is None: 
        return None

    # 2. 【简化且安全的合并逻辑】
    # 直接在它们的DatetimeIndex上进行join，这会自动对齐时间戳。
    past_df = flow_df.join(weather_df, how='inner')

    # 检查join后是否还有24行，以防万一
    if len(past_df) != 24:
        print(f"错误: 合并历史数据后行数不为24 (实际为 {len(past_df)})，请检查数据源。")
        return None

    # 3. 现在 `past_df` 的索引是确定无疑的 DatetimeIndex，可以安全地进行特征工程
    past_df = engineer_features(past_df)

    # ... 后续的归一化和拼接逻辑保持不变 ...
    # 3. 归一化和拼接 (严格对齐训练)
    feature_names = scalers['feature_names']
    
    # 确保所有需要的列都存在于 past_df 中
    required_cols = ['出港流量'] + feature_names['other'] + feature_names['cyclic']
    for col in required_cols:
        if col not in past_df.columns:
             # 如果特征工程后仍然缺少sin/cos列，说明原始列(如wind_dir)就缺失了
             print(f"警告: 合并和特征工程后缺少关键列 '{col}'，将用0填充。")
             past_df[col] = 0
             
    encoder_input_df = past_df[required_cols]

    y_past_scaled = scalers['target'].transform(encoder_input_df[['出港流量']])
    X_other_past_scaled = scalers['other_features'].transform(encoder_input_df[feature_names['other']])
    X_cyclic_past = encoder_input_df[feature_names['cyclic']].values
    
    encoder_data_scaled = np.hstack([y_past_scaled, X_other_past_scaled, X_cyclic_past])
    
    return encoder_data_scaled.reshape(1, 24, -1)


def prepare_decoder_input(prediction_date_str, work_dir, scalers):
    """
    【最终版】准备解码器的输入。
    此版本能处理带有表头、包含'report_day'和'report_time'列的天气预报文件。
    """
    print("\n--- 准备解码器输入 (未来24小时) ---")
    
    # --- 1. 加载并解析带有表头的未来天气预报 ---
    try:
        weather_path = os.path.join(work_dir, 'future_weather_forecast.csv')  
        print(f"正在从 {weather_path} 加载天气预报...")

        # a. 直接使用 read_csv 读取，因为它有表头
        future_weather_df = pd.read_csv(weather_path)

        # b. 解析 'report_day' 和 'report_time' 来构建完整的时间戳
        #    从 prediction_date_str 中获取年份，确保年份正确
        year = datetime.strptime(prediction_date_str, '%Y-%m-%d').year
        
        future_weather_df['timestamp'] = pd.to_datetime(
            str(year) + '-' + future_weather_df['report_day'] + ' ' + future_weather_df['report_time'],
            format='%Y-%m-%d %H:%M',
            errors='coerce' # 如果有格式错误，设为NaT
        )
        future_weather_df.dropna(subset=['timestamp'], inplace=True) # 移除无法解析的行
        
        # c. 将 timestamp 设置为索引，为重采样做准备
        future_weather_df.set_index('timestamp', inplace=True)
        print("天气预报文件加载并解析成功。")
        
    except FileNotFoundError:
        print(f"致命错误: 未来的天气预报文件未找到，请在以下路径创建它:\n{weather_path}")
        return None
    except Exception as e:
        print(f"加载或解析未来天气时出错: {e}")
        return None

    # --- 2. 小时级重采样 (如果需要的话) ---
    # 检查数据是否已经是小时级的，如果不是（比如包含 :30 的记录），则进行重采样
    if any(future_weather_df.index.minute != 0):
        print("检测到数据为30分钟间隔，正在重采样为小时级...")
        weather_hourly_df = future_weather_df.select_dtypes(include=np.number).resample('h').mean()
        print("重采样完成。")
    else:
        print("数据已经是小时级，无需重采样。")
        weather_hourly_df = future_weather_df.select_dtypes(include=np.number)

    # --- 3. 创建标准的24小时DataFrame并合并 ---
    start_dt = datetime.strptime(prediction_date_str, '%Y-%m-%d')
    future_timestamps = pd.date_range(start=start_dt, periods=24, freq='h')
    future_df = pd.DataFrame(index=future_timestamps)

    future_df = future_df.join(weather_hourly_df)
    
    future_df.ffill(inplace=True)
    future_df.bfill(inplace=True)

    if future_df.isnull().values.any():
        print("错误: 合并天气预报后仍有缺失值，请检查预报文件。")
        return None
    
    print("天气预报已成功整合到24小时时间框架中。")

    # --- 4. 特征工程 (逻辑不变) ---
    print("正在为未来数据进行特征工程...")
    future_df = engineer_features(future_df)
    
    # --- 5. 归一化和塑形 (逻辑不变) ---
    feature_names = scalers['feature_names']
    
    required_cols = feature_names['other'] + feature_names['cyclic']
    for col in required_cols:
        if col not in future_df.columns:
            print(f"警告: 特征工程后缺少关键列 '{col}'，将用0填充。")
            future_df[col] = 0

    decoder_input_df = future_df[required_cols]
    
    X_other_future_scaled = scalers['other_features'].transform(decoder_input_df[feature_names['other']])
    X_cyclic_future = decoder_input_df[feature_names['cyclic']].values
    
    decoder_data_scaled = np.hstack([X_other_future_scaled, X_cyclic_future])
    
    # --- 6. 分时段 (逻辑不变) ---
    is_busy_mask = future_df['is_peak_hours'].astype(bool).values
    decoder_input_busy = decoder_data_scaled[is_busy_mask].reshape(1, 16, -1)
    decoder_input_non_busy = decoder_data_scaled[~is_busy_mask].reshape(1, 8, -1)

    print("解码器输入准备完毕。")
    
    return decoder_input_busy, decoder_input_non_busy, is_busy_mask, future_df.index


# ==============================================================================
#  MAIN PREDICTION FUNCTION
# ==============================================================================

def predict_future_flow(model_dir, work_dir, prediction_start_date_str='2024-06-09'):
    print("\n" + "="*60)
    print(f"开始预测 {prediction_start_date_str} 的未来24小时流量...")
    print("="*60)

    # --- 1. 定义和加载核心组件 ---
    model_busy_path = os.path.join(model_dir, 'best_model_busy.h5')
    model_non_busy_path = os.path.join(model_dir, 'best_model_non_busy.h5')
    scaler_path = os.path.join(model_dir, 'zggg_optimized_scalers_new.gz')
    # [TODO] 检查所有文件是否存在

    print("正在加载模型和归一化器...")
    model_busy = load_model(model_busy_path, compile=False)
    model_non_busy = load_model(model_non_busy_path, compile=False)
    with gzip.open(scaler_path, 'rb') as f:
        scalers = joblib.load(f)

    # --- 2. 准备模型输入 ---
    historical_date_str = (datetime.strptime(prediction_start_date_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    last_encoder_input = prepare_encoder_input(historical_date_str, work_dir, scalers)
    decoder_input_busy, decoder_input_non_busy, is_busy_mask, future_timestamps = prepare_decoder_input(prediction_start_date_str, work_dir, scalers)

    if last_encoder_input is None or decoder_input_busy is None:
        print("\n错误：模型输入准备失败，预测中止。")
        return False
        
    # --- 3. 执行模型推理 ---
    print("\n--- 执行模型推理 ---")
    pred_busy_scaled = model_busy.predict([last_encoder_input, decoder_input_busy])
    pred_non_busy_scaled = model_non_busy.predict([last_encoder_input, decoder_input_non_busy])

    # --- 4. 后处理并保存结果 ---
    print("\n--- 后处理并保存结果 ---")
    target_scaler = scalers['target']
    pred_busy_inv = target_scaler.inverse_transform(pred_busy_scaled.flatten().reshape(-1, 1)).flatten()
    pred_non_busy_inv = target_scaler.inverse_transform(pred_non_busy_scaled.flatten().reshape(-1, 1)).flatten()
    
    final_predictions = pd.Series(index=future_timestamps, dtype=float)
    final_predictions.loc[is_busy_mask] = pred_busy_inv
    final_predictions.loc[~is_busy_mask] = pred_non_busy_inv
    final_predictions[final_predictions < 0] = 0
    results_df = pd.DataFrame({'timestamp': final_predictions.index.strftime('%Y-%m-%d %H:%M:%S'), 'predicted_flow': final_predictions.values})
    output_path = os.path.join(work_dir, 'future_24h_prediction.csv')

    results_df.to_csv(output_path, index=False, encoding='utf-8-sig', float_format='%.2f')
    
    print(f"\n预测成功！结果已保存至: {os.path.abspath(output_path)}")
    print("\n--- 预测结果预览 ---")
    print(results_df)
    print("="*60)
    return True

# --- 脚本执行入口 ---
if __name__ == '__main__':
    # [TODO] 设置正确的目录
    MODEL_DIR = r'C:\Users\Administrator\Desktop\final_demo\final_demo\model_1_in\model1_in_output'
    WORK_DIR = r'C:\Users\Administrator\Desktop\final_demo\final_demo\INTERFACE_FINAL\preparation' # 存放备用数据和输出文件的目录
    PREDICTION_DATE = '2025-05-29' # 您想要预测的日期

    # 运行预测
    predict_future_flow(MODEL_DIR, WORK_DIR, PREDICTION_DATE)