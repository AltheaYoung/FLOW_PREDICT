# import pandas as pd
# import numpy as np
# import os
# from sklearn.preprocessing import RobustScaler, StandardScaler
# import joblib
# def create_sliding_windows(data, timestamps, input_seq_len, target_seq_len, target_col_idx=0):
#     """
#     Creates sliding windows for a Seq2Seq model.
#     Also returns the 24-hour target timestamps for each sample.
#     """
#     encoder_input_data = []
#     decoder_input_data = []
#     decoder_target_data = []
#     target_timestamps = [] # 用于存储每个样本对应的24个目标时间戳

#     total_sample_len = input_seq_len + target_seq_len
    
#     for i in range(len(data) - total_sample_len + 1):
#         encoder_input = data[i : i + input_seq_len]
#         decoder_target = data[i + input_seq_len : i + total_sample_len, [target_col_idx]]
#         decoder_future_features = data[i + input_seq_len : i + total_sample_len, :]
#         decoder_input = np.delete(decoder_future_features, target_col_idx, axis=1)
        
#         encoder_input_data.append(encoder_input)
#         decoder_input_data.append(decoder_input)
#         decoder_target_data.append(decoder_target)
        
#         ts_slice = timestamps[i + input_seq_len : i + total_sample_len]
#         target_timestamps.append(ts_slice)
        
#     # 修正：返回正确的 target_timestamps
#     return np.array(encoder_input_data), np.array(decoder_input_data), np.array(decoder_target_data), np.array(target_timestamps)


# def load_and_merge_data(traffic_filepath, weather_dir, airport_code='ZGGG', year='2025'):
#     """
#     Loads traffic (hourly) and weather (30-min) data, resamples weather to hourly, 
#     and then merges them.
#     """
#     print(f"--- Loading and Merging Data for {airport_code} ---")

#     # --- 1. Load and Process 30-min Weather Data ---
#     weather_filename = f"weather_OBCC_{airport_code}.csv"
#     weather_filepath = os.path.join(weather_dir, weather_filename)
    
#     try:
#         weather_df = pd.read_csv(weather_filepath)
#         print(f"Loaded raw weather data with {len(weather_df)} records (30-min interval).")

#         # Parse date and time
#         full_date_str = str(year) + '年' + weather_df['report_day'].astype(str) + ' ' + weather_df['report_time'].astype(str)
#         weather_df['timestamp'] = pd.to_datetime(full_date_str, format='%Y年%m-%d %H:%M')
#         weather_df.set_index('timestamp', inplace=True)
#         weather_df = weather_df.drop(columns=['report_day', 'report_time', 'OBCC'])

#     except Exception as e:
#         print(f"Error processing weather data: {e}")
#         return None

#     # --- 2. Resample Weather Data to Hourly ---
#     print("Resampling weather data to hourly interval...")
    
#     weather_hourly_df = weather_df.select_dtypes(include=np.number).resample('H').mean()
    
#     print(f"Resampling complete. Weather data now has {len(weather_hourly_df)} records (hourly).")
    
#     # --- 3. Load Hourly Traffic Data ---
#     try:
#         traffic_df = pd.read_csv(traffic_filepath)
#         traffic_df = traffic_df[traffic_df['地点'] == airport_code].copy()
#         if traffic_df.empty:
#             raise ValueError(f"No traffic data found for airport code '{airport_code}'")
            
#         traffic_df['timestamp'] = pd.to_datetime(traffic_df['小时'])
#         traffic_df = traffic_df.drop(columns=['地点', '小时'])
#         print(f"Loaded {len(traffic_df)} hourly traffic records.")

#     except Exception as e:
#         print(f"Error loading traffic data: {e}")
#         return None

#     # --- 4. Merge Data ---
#     merged_df = pd.merge(
#         traffic_df,          
#         weather_hourly_df,   
#         left_on='timestamp', 
#         right_index=True,    
#         how='inner'          
#     )
#     merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)

#     print(f"Successfully merged data. Final shape: {merged_df.shape}")
#     print("Merged DataFrame columns:", merged_df.columns.tolist())
    
#     return merged_df


# def feature_engineer_and_select(df):
#     """
#     Performs feature engineering (time features) and selects final columns.
#     """
#     print("\n--- Performing Feature Engineering ---")
    
#     # Create time features
#     df['hour'] = df['timestamp'].dt.hour
#     df['dayofweek'] = df['timestamp'].dt.dayofweek # Monday=0, Sunday=6
    
#     # *** NEW FEATURE: Create a binary feature for peak operating hours (7:00 - 23:00) ***
#     # This creates a boolean Series (True/False) and converts it to integers (1/0)
#     df['is_peak_hours'] = ((df['hour'] >= 7) & (df['hour'] <= 23)).astype(int)
#     print("Created new feature 'is_peak_hours'.")

#     # Encode cyclical time features
#     df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23.0)
#     df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23.0)
#     df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 6.0)
#     df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 6.0)

#     # Select final features
#     target_col = '进港流量'
    
#     # *** MODIFIED: Add the new feature to the list of features to be used ***
#     feature_cols = [
#         'temp_c', 'wind_dir_sin', 'wind_dir_cos','wind_speed_mps', 'visibility_m',
#         'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
#         'pressure_hpa','dewpoint_c','wind_gust_mps',
#         'is_peak_hours'  # <-- Added here
#     ]
    
#     # Ensure all selected columns exist
#     final_cols = [target_col] + [col for col in feature_cols if col in df.columns]
    
#     # Handle missing columns
#     missing_cols = [col for col in feature_cols if col not in df.columns]
#     if missing_cols:
#         print(f"Warning: The following feature columns were not found and will be ignored: {missing_cols}")

#     final_df = df[final_cols].copy()
    
#     # Handle potential missing values
#     final_df.ffill(inplace=True)
#     final_df.bfill(inplace=True)

#     print(f"Selected {len(final_df.columns)} final columns for the model.")
#     print("Final columns (target is first):", final_df.columns.tolist())
    
#     return final_df



# def combine_features(other_scaled, cyclic):
#     # 确保索引一致
#     cyclic_np = cyclic.values
#     # 合并特征
#     return np.hstack([other_scaled, cyclic_np])
# def split_features_and_target(df):
#     """
#     分离特征和目标变量。
#     将特征分为两组：需要缩放的(other)和不需要缩放的(passthrough)。
#     """
#     target_col = '进港流量'
    
#     # *** MODIFIED: Renamed list for clarity and added the new feature ***
#     # 这些特征已经处理到[-1,1]或[0,1]范围，不需要RobustScaler再次处理
#     passthrough_features = [
#         'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 
#         'wind_dir_sin', 'wind_dir_cos',
#         'is_peak_hours' # <-- Added here
#     ]
    
#     # 其他需要归一化的特征
#     other_features = [col for col in df.columns if col != target_col and col not in passthrough_features]
    
#     # 确保我们没有遗漏任何特征
#     found_passthrough = [f for f in passthrough_features if f in df.columns]

#     print(f"Splitting features: {len(other_features)} to be scaled, {len(found_passthrough)} to pass through.")
    
#     return df[other_features], df[found_passthrough], df[[target_col]]


# # --- Main Execution ---
# if __name__ == '__main__':
#     # --- Configuration ---
#     TARGET_AIRPORT = 'ZGGG' # Guangzhou Baiyun International Airport
    
#     # Define your data paths
#     TRAFFIC_DATA_PATH = r'C:\Users\Administrator\Desktop\Q1\in\data\hourly_flow_statistics.csv'
#     WEATHER_DATA_DIR = r'C:\Users\Administrator\Desktop\Q1\in\data'
#     OUTPUT_DIR = r'C:\Users\Administrator\Desktop\Q1\in'
    
#     # Model sequence lengths
#     INPUT_SEQ_LEN = 24  # Use past 48 hours of data
#     TARGET_SEQ_LEN = 24 # Predict next 24 hours
    
#     # 1. Load and merge the raw data
#     merged_data = load_and_merge_data(TRAFFIC_DATA_PATH, WEATHER_DATA_DIR, TARGET_AIRPORT)
    
#     if merged_data is not None:
#         # 2. Engineer features
#         feature_df = feature_engineer_and_select(merged_data)

        
#         # --- 全新逻辑从这里开始 ---

#         # 提取完整的时间戳列表，供后续使用
#         all_timestamps = merged_data['timestamp'].values
        
#         # 3. 在整个数据集上进行归一化
#         print("\n--- Normalizing the ENTIRE dataset before windowing ---")
#         X_other_all, X_cyclic_all, y_all = split_features_and_target(feature_df)

        
#         # 创建并拟合 Scaler
#         scaler_other = RobustScaler()
#         scaler_target = RobustScaler()
        
#         X_other_scaled_all = scaler_other.fit_transform(X_other_all)
#         y_scaled_all = scaler_target.fit_transform(y_all)
        
#         # 合并所有特征
#         X_scaled_all = combine_features(X_other_scaled_all, X_cyclic_all)
        
#         # 组合特征和目标，准备创建窗口
#         all_scaled_data = np.hstack([y_scaled_all, X_scaled_all])
        
#         # 4. 在完整的、归一化后的数据上创建所有可能的滑动窗口样本
#         print("\n--- Creating sliding windows on the full dataset ---")
#         encoder_all, decoder_all, target_all, timestamps_all = create_sliding_windows(
#             all_scaled_data, 
#             all_timestamps, # 传入完整的时间戳列表
#             INPUT_SEQ_LEN, 
#             TARGET_SEQ_LEN
#         )
        
#         # 5. 按时间顺序划分已经配对好的样本
#         print("\n--- Splitting SAMPLES chronologically (2:1 ratio) ---")
#         n_samples = len(encoder_all)
#         train_size = int(n_samples * 0.95)
        
#         # 划分训练集
#         X_train_encoder = encoder_all[:train_size]
#         X_train_decoder = decoder_all[:train_size]
#         y_train_window = target_all[:train_size]
#         train_timestamps = timestamps_all[:train_size]
        
#         # 划分验证集
#         X_val_encoder = encoder_all[train_size:]
#         X_val_decoder = decoder_all[train_size:]
#         y_val_window = target_all[train_size:]
#         val_timestamps = timestamps_all[train_size:]

#         # （可选）如果你需要测试集，可以按 6:2:2 等比例划分
#         # train_size = int(n_samples * 0.6)
#         # val_size = int(n_samples * 0.2)
#         # ...
        
#         print(f"Total samples: {n_samples}")
#         print(f"Train samples: {len(X_train_encoder)}, from {train_timestamps[0]} to {train_timestamps[-1]}")
#         print(f"Validation samples: {len(X_val_encoder)}, from {val_timestamps[0]} to {val_timestamps[-1]}")
        
#         # 6. 保存所有需要的文件
#         print("\n--- Saving processed data and scalers ---")
        
#         # 保存Scaler
#         scalers = {
#             'other_features': scaler_other,
#             'target': scaler_target,
#             'feature_names': { 'other': X_other_all.columns.tolist(), 'cyclic': X_cyclic_all.columns.tolist() }
#         }
#         scaler_path = os.path.join(OUTPUT_DIR, 'zggg_optimized_scalers.gz')
#         joblib.dump(scalers, scaler_path)
#         print(f"Scalers saved to {scaler_path}")
        
#         # 保存处理好的数据数组，包含正确的时间戳
#         np.savez(os.path.join(OUTPUT_DIR, 'processed_zggg_data_optimized.npz'),
#                  X_train_encoder=X_train_encoder, X_train_decoder=X_train_decoder, y_train=y_train_window, train_timestamps=train_timestamps,
#                  X_val_encoder=X_val_encoder, X_val_decoder=X_val_decoder, y_val=y_val_window, val_timestamps=val_timestamps
#                 )
#         print(f"Processed data saved to {os.path.join(OUTPUT_DIR, 'processed_zggg_data_optimized.npz')}")

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib

def create_sliding_windows(data, timestamps, input_seq_len, target_seq_len, target_col_idx=0):
    """
    Creates sliding windows for a Seq2Seq model.
    Also returns the 24-hour target timestamps for each sample.
    """
    encoder_input_data = []
    decoder_input_data = []
    decoder_target_data = []
    target_timestamps = [] # 用于存储每个样本对应的24个目标时间戳

    total_sample_len = input_seq_len + target_seq_len
    
    for i in range(len(data) - total_sample_len + 1):
        encoder_input = data[i : i + input_seq_len]
        decoder_target = data[i + input_seq_len : i + total_sample_len, [target_col_idx]]
        decoder_future_features = data[i + input_seq_len : i + total_sample_len, :]
        decoder_input = np.delete(decoder_future_features, target_col_idx, axis=1)
        
        encoder_input_data.append(encoder_input)
        decoder_input_data.append(decoder_input)
        decoder_target_data.append(decoder_target)
        
        ts_slice = timestamps[i + input_seq_len : i + total_sample_len]
        target_timestamps.append(ts_slice)
        
    # 修正：返回正确的 target_timestamps
    return np.array(encoder_input_data), np.array(decoder_input_data), np.array(decoder_target_data), np.array(target_timestamps)


def load_and_merge_data(traffic_filepath, weather_dir, airport_code='ZGGG', year='2025'):
    """
    Loads traffic (hourly) and weather (30-min) data, resamples weather to hourly, 
    and then merges them.
    """
    print(f"--- Loading and Merging Data for {airport_code} ---")

    # --- 1. Load and Process 30-min Weather Data ---
    weather_filename = f"weather_OBCC_{airport_code}.csv"
    weather_filepath = os.path.join(weather_dir, weather_filename)
    
    try:
        weather_df = pd.read_csv(weather_filepath)
        print(f"Loaded raw weather data with {len(weather_df)} records (30-min interval).")

        # Parse date and time
        full_date_str = str(year) + '年' + weather_df['report_day'].astype(str) + ' ' + weather_df['report_time'].astype(str)
        weather_df['timestamp'] = pd.to_datetime(full_date_str, format='%Y年%m-%d %H:%M')
        weather_df.set_index('timestamp', inplace=True)
        weather_df = weather_df.drop(columns=['report_day', 'report_time', 'OBCC'])

    except Exception as e:
        print(f"Error processing weather data: {e}")
        return None

    # --- 2. Resample Weather Data to Hourly ---
    print("Resampling weather data to hourly interval...")
    
    weather_hourly_df = weather_df.select_dtypes(include=np.number).resample('H').mean()
    
    print(f"Resampling complete. Weather data now has {len(weather_hourly_df)} records (hourly).")
    
    # --- 3. Load Hourly Traffic Data ---
    try:
        traffic_df = pd.read_csv(traffic_filepath)
        traffic_df = traffic_df[traffic_df['地点'] == airport_code].copy()
        if traffic_df.empty:
            raise ValueError(f"No traffic data found for airport code '{airport_code}'")
            
        traffic_df['timestamp'] = pd.to_datetime(traffic_df['小时'])
        traffic_df = traffic_df.drop(columns=['地点', '小时'])
        print(f"Loaded {len(traffic_df)} hourly traffic records.")

    except Exception as e:
        print(f"Error loading traffic data: {e}")
        return None

    # --- 4. Merge Data ---
    merged_df = pd.merge(
        traffic_df,          
        weather_hourly_df,   
        left_on='timestamp', 
        right_index=True,    
        how='inner'          
    )
    merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)

    print(f"Successfully merged data. Final shape: {merged_df.shape}")
    print("Merged DataFrame columns:", merged_df.columns.tolist())
    
    return merged_df


def feature_engineer_and_select(df):
    """
    Performs feature engineering (time features) and selects final columns.
    """
    print("\n--- Performing Feature Engineering ---")
    
    # Create time features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek # Monday=0, Sunday=6
    
    # *** NEW FEATURE: Create a binary feature for peak operating hours (7:00 - 23:00) ***
    # This creates a boolean Series (True/False) and converts it to integers (1/0)
    df['is_peak_hours'] = ((df['hour'] >= 7) & (df['hour'] <= 23)).astype(int)
    print("Created new feature 'is_peak_hours'.")

    # Encode cyclical time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 6.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 6.0)

    # Select final features
    target_col = '出港流量'
    
    # *** MODIFIED: Add the new feature to the list of features to be used ***
    feature_cols = [
        'temp_c', 'wind_dir_sin', 'wind_dir_cos','wind_speed_mps', 'visibility_m',
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
        'pressure_hpa','dewpoint_c','wind_gust_mps',
        'is_peak_hours'  # <-- Added here
    ]
    
    # Ensure all selected columns exist
    final_cols = [target_col] + [col for col in feature_cols if col in df.columns]
    
    # Handle missing columns
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following feature columns were not found and will be ignored: {missing_cols}")

    final_df = df[final_cols].copy()
    
    # Handle potential missing values
    final_df.ffill(inplace=True)
    final_df.bfill(inplace=True)

    print(f"Selected {len(final_df.columns)} final columns for the model.")
    print("Final columns (target is first):", final_df.columns.tolist())
    
    return final_df



def combine_features(other_scaled, cyclic):
    # 确保索引一致
    cyclic_np = cyclic.values
    # 合并特征
    return np.hstack([other_scaled, cyclic_np])

def split_features_and_target(df):
    """
    分离特征和目标变量。
    将特征分为两组：需要缩放的(other)和不需要缩放的(passthrough)。
    """
    target_col = '出港流量'
    
    # *** MODIFIED: Renamed list for clarity and added the new feature ***
    # 这些特征已经处理到[-1,1]或[0,1]范围，不需要RobustScaler再次处理
    passthrough_features = [
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 
        'wind_dir_sin', 'wind_dir_cos',
        'is_peak_hours' # <-- Added here
    ]
    
    # 其他需要归一化的特征
    other_features = [col for col in df.columns if col != target_col and col not in passthrough_features]
    
    # 确保我们没有遗漏任何特征
    found_passthrough = [f for f in passthrough_features if f in df.columns]

    print(f"Splitting features: {len(other_features)} to be scaled, {len(found_passthrough)} to pass through.")
    
    return df[other_features], df[found_passthrough], df[[target_col]]


# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    TARGET_AIRPORT = 'ZGGG' # Guangzhou Baiyun International Airport
    
    # Define your data paths
    TRAFFIC_DATA_PATH = r'C:\Users\Administrator\Desktop\Q1\Q1_OUT_LAST\Q1_OUT_LAST\hourly_flow_statistics.csv'
    WEATHER_DATA_DIR = r'C:\Users\Administrator\Desktop\Q1\Q1_OUT_LAST\Q1_OUT_LAST\DIV_OBCC(1)\DIV_OBCC'
    OUTPUT_DIR = r'C:\Users\Administrator\Desktop\Q1\Q1_OUT_LAST\Q1_OUT_LAST'
    
    # Model sequence lengths
    INPUT_SEQ_LEN = 24  # Use past 24 hours of data
    TARGET_SEQ_LEN = 24 # Predict next 24 hours
    
    # 1. Load and merge the raw data
    merged_data = load_and_merge_data(TRAFFIC_DATA_PATH, WEATHER_DATA_DIR, TARGET_AIRPORT)
    
    if merged_data is not None:
        # 2. Engineer features
        feature_df = feature_engineer_and_select(merged_data)
        
        # 提取完整的时间戳列表，供后续使用
        all_timestamps = merged_data['timestamp'].values
        
        # 3. 在整个数据集上进行归一化
        print("\n--- Normalizing the ENTIRE dataset before windowing ---")
        X_other_all, X_cyclic_all, y_all = split_features_and_target(feature_df)

        # 创建并拟合 Scaler
        scaler_other = RobustScaler()
        scaler_target = RobustScaler()
        
        X_other_scaled_all = scaler_other.fit_transform(X_other_all)
        y_scaled_all = scaler_target.fit_transform(y_all)
        
        # 合并所有特征
        X_scaled_all = combine_features(X_other_scaled_all, X_cyclic_all)
        
        # 组合特征和目标，准备创建窗口
        all_scaled_data = np.hstack([y_scaled_all, X_scaled_all])
        
        # 4. 在完整的、归一化后的数据上创建所有可能的滑动窗口样本
        print("\n--- Creating sliding windows on the full dataset ---")
        encoder_all, decoder_all, target_all, timestamps_all = create_sliding_windows(
            all_scaled_data, 
            all_timestamps, # 传入完整的时间戳列表
            INPUT_SEQ_LEN, 
            TARGET_SEQ_LEN
        )
        
        # =================================================================================
        # === MODIFIED SECTION: REPLACED PERCENTAGE SPLIT WITH DATE-BASED SPLIT         ===
        # =================================================================================
        
        # 5. 按指定日期划分训练集与验证集 (测试集)
        print("\n--- Splitting SAMPLES by specified date range ---")

        # 从所有样本中提取每个样本的第一个目标时间戳，用于定位
        # `timestamps_all` 的形状是 (n_samples, 24)，我们只关心每个样本的起始预测时间
        start_timestamps = timestamps_all[:, 0]

        # 定义日期边界 (假设您的数据年份是2025)
        # 注意：需要使用 numpy 的 datetime64 类型进行比较
        train_start_date = np.datetime64('2025-05-01T00:00:00')
        val_start_date = np.datetime64('2025-05-31T00:00:00')
        val_end_date = np.datetime64('2025-06-01T00:00:00') # 验证集的结束边界 (不包含)

        try:
            # 使用 np.searchsorted 高效查找索引
            # 它会找到第一个时间戳 >= 我们指定日期 的位置
            train_start_index = np.searchsorted(start_timestamps, train_start_date, side='left')
            val_start_index = np.searchsorted(start_timestamps, val_start_date, side='left')
            val_end_index = np.searchsorted(start_timestamps, val_end_date, side='left')

            if train_start_index >= val_start_index or val_start_index >= val_end_index:
                raise ValueError("Date ranges are incorrect or no data found for the specified periods.")

            # 根据找到的索引切分所有数据数组
            # 训练集: [train_start_index, val_start_index)
            X_train_encoder = encoder_all[train_start_index:val_start_index]
            X_train_decoder = decoder_all[train_start_index:val_start_index]
            y_train_window = target_all[train_start_index:val_start_index]
            train_timestamps = timestamps_all[train_start_index:val_start_index]

            # 验证集 (测试集): [val_start_index, val_end_index)
            X_val_encoder = encoder_all[val_start_index:val_end_index]
            X_val_decoder = decoder_all[val_start_index:val_end_index]
            y_val_window = target_all[val_start_index:val_end_index]
            val_timestamps = timestamps_all[val_start_index:val_end_index]
            
            print(f"Date-based splitting successful.")
            
            # 安全地打印信息，防止因找不到数据而引发索引错误
            if len(X_train_encoder) > 0:
                print(f"Train samples: {len(X_train_encoder)}, covering predictions from {train_timestamps[0, 0]} to {train_timestamps[-1, -1]}")
            else:
                print("Warning: No training samples found for the period May 1 to May 30.")
                
            if len(X_val_encoder) > 0:
                print(f"Validation samples: {len(X_val_encoder)}, covering predictions from {val_timestamps[0, 0]} to {val_timestamps[-1, -1]}")
            else:
                print("Warning: No validation samples found for May 31.")

        except (ValueError, IndexError) as e:
            print(f"Error during data splitting: {e}")
            print("No data will be saved. Please check your date ranges and input data.")
            # 在这种情况下，后续的保存步骤可能会出错，将数组设置为空
            X_train_encoder, X_train_decoder, y_train_window, train_timestamps = (np.array([]),)*4
            X_val_encoder, X_val_decoder, y_val_window, val_timestamps = (np.array([]),)*4

        # =================================================================================
        # === END OF MODIFIED SECTION                                                   ===
        # =================================================================================

        # 6. 保存所有需要的文件
        # 仅在成功创建数据后才进行保存
        if len(X_train_encoder) > 0 and len(X_val_encoder) > 0:
            print("\n--- Saving processed data and scalers ---")
            
            # 保存Scaler
            scalers = {
                'other_features': scaler_other,
                'target': scaler_target,
                'feature_names': { 'other': X_other_all.columns.tolist(), 'cyclic': X_cyclic_all.columns.tolist() }
            }
            scaler_path = os.path.join(OUTPUT_DIR, 'zggg_optimized_scalers_new.gz')
            joblib.dump(scalers, scaler_path)
            print(f"Scalers saved to {scaler_path}")
            
            # 保存处理好的数据数组，包含正确的时间戳
            output_filepath = os.path.join(OUTPUT_DIR, 'processed_zggg_data_optimized_new.npz')
            np.savez(output_filepath,
                     X_train_encoder=X_train_encoder, X_train_decoder=X_train_decoder, y_train=y_train_window, train_timestamps=train_timestamps,
                     X_val_encoder=X_val_encoder, X_val_decoder=X_val_decoder, y_val=y_val_window, val_timestamps=val_timestamps
                    )
            print(f"Processed data saved to {output_filepath}")
        else:
            print("\n--- Skipping save process due to empty datasets from splitting. ---")