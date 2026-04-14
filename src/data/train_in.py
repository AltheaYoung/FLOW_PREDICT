# # -*- coding: utf-8 -*-
# import numpy as np
# import os
# import pandas as pd
# import gzip
# from joblib import load
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from keras.models import Model, load_model
# from keras.layers import Input, LSTM, Dense, Concatenate, Bidirectional, Dropout, TimeDistributed, AdditiveAttention
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tqdm import tqdm

# # ==============================================================================
# # 1. 模型构建与辅助函数 (这部分无需改动，与之前相同)
# # ==============================================================================
# def build_seq2seq_attention_model(input_seq_len, target_seq_len, num_encoder_features, num_decoder_features,
#                                   lstm_units=128, dropout_rate=0.4):
#     """通用的Seq2Seq模型构建函数。"""
#     encoder_inputs = Input(shape=(input_seq_len, num_encoder_features), name='encoder_inputs')
#     encoder_lstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(0.001)), name='encoder_bi_lstm')(encoder_inputs)
#     encoder_outputs1 = Dropout(dropout_rate)(encoder_lstm1)
#     encoder_lstm2_layer = LSTM(lstm_units, return_sequences=True, return_state=True, name='encoder_lstm_2')
#     encoder_outputs2, state_h, state_c = encoder_lstm2_layer(encoder_outputs1)
#     encoder_states = [state_h, state_c]

#     decoder_inputs = Input(shape=(target_seq_len, num_decoder_features), name='decoder_inputs')
#     decoder_lstm = LSTM(lstm_units, return_sequences=True, name='decoder_lstm')(decoder_inputs, initial_state=encoder_states)

#     attention_layer = AdditiveAttention(name='attention_layer')
#     context_vector = attention_layer([decoder_lstm, encoder_outputs2])

#     decoder_combined_context = Concatenate(axis=-1, name='concat_decoder_attention')([decoder_lstm, context_vector])
#     decoder_combined_context = Dropout(dropout_rate)(decoder_combined_context)

#     dense_hidden = TimeDistributed(Dense(64, activation='relu', kernel_regularizer=l2(0.001)), name='output_dense_hidden')(decoder_combined_context)
#     outputs_hidden = Dropout(dropout_rate)(dense_hidden)
#     dense_output = TimeDistributed(Dense(1, activation='linear'), name='output_dense_final')(outputs_hidden)

#     model = Model([encoder_inputs, decoder_inputs], dense_output)
#     optimizer = Adam(learning_rate=0.001)
#     model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
#     return model

# def plot_history(history, save_path):
#     """绘制训练历史并保存。"""
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['mae'], label='Training MAE')
#     plt.plot(history.history['val_mae'], label='Validation MAE')
#     plt.title('Training and Validation MAE')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"训练历史图表已保存至 {save_path}")

# def save_hourly_predictions(predictions, true_values, target_timestamps, scaler, save_path):
#     """使用精确的时间戳数组保存小时级预测（此函数现在更健壮）。"""
#     assert predictions.shape[0:2] == true_values.shape[0:2] == target_timestamps.shape[0:2], "Shape mismatch!"

#     n_samples, n_timesteps, _ = predictions.shape
#     predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(n_samples, n_timesteps, 1)
#     true_values_inv = scaler.inverse_transform(true_values.reshape(-1, 1)).reshape(n_samples, n_timesteps, 1)

#     results = []
#     for i in range(n_samples):
#         for j in range(n_timesteps):
#             results.append({
#                 'timestamp': pd.to_datetime(target_timestamps[i, j]).strftime('%Y-%m-%d %H:%M:%S'),
#                 'predicted_flow': predictions_inv[i, j, 0],
#                 'actual_flow': true_values_inv[i, j, 0]
#             })

#     results_df = pd.DataFrame(results).sort_values('timestamp')
#     results_df.to_csv(save_path, index=False, encoding='utf-8-sig', float_format='%.2f')
#     print(f"小时级预测结果已保存至: {save_path}")

# def aggregate_hourly_predictions(input_csv_path, output_csv_path):
#     """聚合小时级预测结果。"""
#     print(f"\n聚合预测结果: {input_csv_path}")
#     try:
#         df = pd.read_csv(input_csv_path)
#     except FileNotFoundError:
#         print(f"警告: 文件未找到，跳过聚合: {input_csv_path}")
#         return

#     aggregated_df = df.groupby('timestamp').agg({
#         'actual_flow': 'first',
#         'predicted_flow': ['mean', 'size']
#     })
#     # 重命名列
#     aggregated_df.columns = ['actual_flow', 'predicted_flow_mean', 'prediction_count']
#     aggregated_df = aggregated_df.reset_index()

#     aggregated_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig', float_format='%.2f')
#     print(f"聚合后的预测已保存至: {output_csv_path}")


# # ==============================================================================
# # 2. 主执行程序 (这部分无需改动，与之前相同)
# # ==============================================================================
# if __name__ == '__main__':
#     # --- 步骤 1: 基础配置 ---
#     DATA_PATH = r'C:\Users\24766\Desktop\final_work\final_demo\final_demo\final_demo\INTERFACE_FINAL\preparation_in\zggg_optimized_scalers_new.gz'
#     SCALER_PATH = r'C:\Users\24766\Desktop\final_work\final_demo\final_demo\final_demo\INTERFACE_FINAL\preparation_in\zggg_optimized_scalers_new.gz'
#     MODEL_SAVE_DIR = r'C:\Users\24766\Desktop\final_work\final_demo\final_demo\final_demo\INTERFACE_FINAL\model_output_dual_corrected'
#     os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

#     # --- 步骤 2: 定义时段判断函数 ---
#     def is_busy(hour):
#         return 7 <= hour <= 22

#     training_configs = [
#         {"name": "busy", "period_checker": is_busy, "target_seq_len": 16},
#         {"name": "non_busy", "period_checker": lambda h: not is_busy(h), "target_seq_len": 8}
#     ]

#     # --- 步骤 3: 加载完整数据 ---
#     print("="*60)
#     print("加载完整数据集...")
#     with np.load(DATA_PATH, allow_pickle=True) as data:
#         X_train_encoder = data['X_train_encoder']
#         X_train_decoder_full = data['X_train_decoder']
#         y_train_full = data['y_train']
#         train_timestamps_full = data['train_timestamps']

#         X_val_encoder = data['X_val_encoder']
#         X_val_decoder_full = data['X_val_decoder']
#         y_val_full = data['y_val']
#         val_timestamps_full = data['val_timestamps']

#     print("加载归一化器...")
#     with gzip.open(SCALER_PATH, 'rb') as f:
#         scaler = load(f)
#     target_scaler = scaler['target']
#     print("数据和归一化器加载完毕。")

#     # --- 步骤 4: 循环训练每个模型 (此部分逻辑不变) ---
#     for config in training_configs:
#         period_name = config["name"]
#         period_checker = config["period_checker"]
#         target_seq_len = config["target_seq_len"]

#         print("\n" + "="*60)
#         print(f"开始处理【{period_name.upper()}】时段模型")
#         print("="*60)

#         # 4a. 动态切片数据
#         print(f"正在为【{period_name}】时段动态准备数据...")

#         # --- 为训练集动态切片 ---
#         train_indices_list = [
#             [j for j, ts in enumerate(sample_ts) if period_checker(pd.to_datetime(ts).hour)]
#             for sample_ts in tqdm(train_timestamps_full, desc=f"Scanning Train Timestamps for '{period_name}'")
#         ]
#         assert all(len(indices) == target_seq_len for indices in train_indices_list), f"Train data error: Not all samples have {target_seq_len} '{period_name}' hours!"

#         y_train_period = np.array([y_train_full[i, indices, :] for i, indices in enumerate(train_indices_list)])
#         X_train_decoder_period = np.array([X_train_decoder_full[i, indices, :] for i, indices in enumerate(train_indices_list)])
#         train_timestamps_period = np.array([train_timestamps_full[i, indices] for i, indices in enumerate(train_indices_list)])

#         # --- 为验证集动态切片 ---
#         val_indices_list = [
#             [j for j, ts in enumerate(sample_ts) if period_checker(pd.to_datetime(ts).hour)]
#             for sample_ts in tqdm(val_timestamps_full, desc=f"Scanning Val Timestamps for '{period_name}'")
#         ]
#         assert all(len(indices) == target_seq_len for indices in val_indices_list), f"Validation data error: Not all samples have {target_seq_len} '{period_name}' hours!"

#         y_val_period = np.array([y_val_full[i, indices, :] for i, indices in enumerate(val_indices_list)])
#         X_val_decoder_period = np.array([X_val_decoder_full[i, indices, :] for i, indices in enumerate(val_indices_list)])
#         val_timestamps_period = np.array([val_timestamps_full[i, indices] for i, indices in enumerate(val_indices_list)])

#         print(f"动态切片完成。训练目标y形状: {y_train_period.shape}, 验证目标y形状: {y_val_period.shape}")

#         # 4b. 构建模型
#         model = build_seq2seq_attention_model(
#             input_seq_len=X_train_encoder.shape[1],
#             target_seq_len=target_seq_len,
#             num_encoder_features=X_train_encoder.shape[2],
#             num_decoder_features=X_train_decoder_period.shape[2]
#         )

#         # 4c. 训练模型
#         best_model_path = os.path.join(MODEL_SAVE_DIR, f'best_model_{period_name}.h5')
#         checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
#         early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True, verbose=1)

#         print(f"\n开始训练【{period_name}】模型...")
#         history = model.fit(
#             [X_train_encoder, X_train_decoder_period], y_train_period,
#             epochs=100, batch_size=32,
#             validation_data=([X_val_encoder, X_val_decoder_period], y_val_period),
#             callbacks=[checkpoint, early_stopping]
#         )

#         # 4d. 保存和评估
#         print(f"【{period_name}】模型训练完成。")
#         plot_history(history, os.path.join(MODEL_SAVE_DIR, f'training_history_{period_name}_new.png'))

#         best_model = load_model(best_model_path)
#         val_predictions = best_model.predict([X_val_encoder, X_val_decoder_period])

#         val_hourly_path = os.path.join(MODEL_SAVE_DIR, f'validation_hourly_predictions_{period_name}_new.csv')

#         save_hourly_predictions(val_predictions, y_val_period, val_timestamps_period, target_scaler, val_hourly_path)
        
#         # 【注意】此处聚合函数内的列名是'predicted_flow_mean'，在后面合并时需要使用这个列名
#         val_agg_path = os.path.join(MODEL_SAVE_DIR, f'validation_aggregated_predictions_{period_name}_new.csv')
#         aggregate_hourly_predictions(val_hourly_path, val_agg_path)


#     # ==============================================================================
#     # 5. 【重大修改】合并两个模型的预测结果，并择优选择
#     # ==============================================================================
#     print("\n" + "="*60)
#     print("择优合并所有时段的预测结果...")
#     try:
#         # 5a. 加载两个模型聚合后的预测结果
#         busy_pred_path = os.path.join(MODEL_SAVE_DIR, 'validation_aggregated_predictions_busy_new.csv')
#         non_busy_pred_path = os.path.join(MODEL_SAVE_DIR, 'validation_aggregated_predictions_non_busy_new.csv')

#         df_busy = pd.read_csv(busy_pred_path)
#         df_non_busy = pd.read_csv(non_busy_pred_path)

#         # 为了区分来源，我们重命名预测列
#         df_busy.rename(columns={'predicted_flow_mean': 'predicted_flow_busy'}, inplace=True)
#         df_non_busy.rename(columns={'predicted_flow_mean': 'predicted_flow_non_busy'}, inplace=True)

#         # 5b. 使用外连接(outer merge)合并两个DataFrame，确保所有时间戳都被包含
#         # 使用 'timestamp' 和 'actual_flow'作为键。因为对于同一个时间点，实际流量(actual_flow)是唯一的
#         df_merged = pd.merge(
#             df_busy[['timestamp', 'actual_flow', 'predicted_flow_busy']],
#             df_non_busy[['timestamp', 'actual_flow', 'predicted_flow_non_busy']],
#             on=['timestamp', 'actual_flow'],
#             how='outer'
#         )

#         # 5c. 计算每个模型预测的绝对误差
#         # .abs() 用于取绝对值
#         df_merged['error_busy'] = (df_merged['predicted_flow_busy'] - df_merged['actual_flow']).abs()
#         df_merged['error_non_busy'] = (df_merged['predicted_flow_non_busy'] - df_merged['actual_flow']).abs()

#         # 5d. 核心逻辑：择优选择
#         # 使用 np.where 来实现条件选择
#         # 条件：当 'error_busy' 小于 'error_non_busy' 时，选择繁忙时段的预测值。
#         #       如果 'error_busy' 不是更小 (即更大、相等或其中一个为NaN)，则选择非繁忙时段的预测值。
#         #       我们使用 fillna(np.inf) 来处理一个模型没有预测值的情况，
#         #       确保有预测值的模型总是被优先选择。
#         df_merged['final_prediction'] = np.where(
#             df_merged['error_busy'].fillna(np.inf) < df_merged['error_non_busy'].fillna(np.inf),
#             df_merged['predicted_flow_busy'],
#             df_merged['predicted_flow_non_busy']
#         )
        
#         # 挑选出最终需要的列
#         df_final = df_merged[['timestamp', 'actual_flow', 'final_prediction']]
#         df_final = df_final.rename(columns={'final_prediction': 'predicted_flow'}) # 重命名回标准列名

#         # 5e. 保存最终的、经过优选的预测结果
#         df_final = df_final.sort_values('timestamp')
#         combined_path = os.path.join(MODEL_SAVE_DIR, 'validation_FINAL_optimal_predictions_new.csv')

#         df_final.to_csv(combined_path, index=False, encoding='utf-8-sig', float_format='%.2f')
#         print(f"成功择优合并！最终优化预测文件已保存至: {combined_path}")

#     except FileNotFoundError as e:
#         print(f"警告: 文件未找到: {e.filename}。跳过最终合并步骤。")
#     except Exception as e:
#         print(f"合并过程中发生未知错误: {e}")

#     print("\n--- 所有流程执行完毕 ---")

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import gzip
from joblib import load
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Concatenate, Bidirectional, Dropout, TimeDistributed, AdditiveAttention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau  # 新增学习率衰减回调
from tqdm import tqdm

# ==============================================================================
# 1. 模型构建与辅助函数 (增强正则化+优化可视化)
# ==============================================================================
def build_seq2seq_attention_model(input_seq_len, target_seq_len, num_encoder_features, num_decoder_features,
                                  lstm_units=128, dropout_rate=0.5):  # 提高Dropout率至0.5
    """增强正则化的Seq2Seq模型，缓解过拟合"""
    encoder_inputs = Input(shape=(input_seq_len, num_encoder_features), name='encoder_inputs')
    # 增大L2正则强度至0.005，限制权重过大
    encoder_lstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(0.005)), name='encoder_bi_lstm')(encoder_inputs)
    encoder_outputs1 = Dropout(dropout_rate)(encoder_lstm1)
    # 编码器第二层LSTM添加L2正则
    encoder_lstm2_layer = LSTM(lstm_units, return_sequences=True, return_state=True, kernel_regularizer=l2(0.005), name='encoder_lstm_2')
    encoder_outputs2, state_h, state_c = encoder_lstm2_layer(encoder_outputs1)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(target_seq_len, num_decoder_features), name='decoder_inputs')
    # 解码器LSTM添加L2正则
    decoder_lstm = LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(0.005), name='decoder_lstm')(decoder_inputs, initial_state=encoder_states)

    attention_layer = AdditiveAttention(name='attention_layer')
    context_vector = attention_layer([decoder_lstm, encoder_outputs2])

    decoder_combined_context = Concatenate(axis=-1, name='concat_decoder_attention')([decoder_lstm, context_vector])
    decoder_combined_context = Dropout(dropout_rate)(decoder_combined_context)

    # 隐藏层添加L2正则
    dense_hidden = TimeDistributed(Dense(64, activation='relu', kernel_regularizer=l2(0.005)), name='output_dense_hidden')(decoder_combined_context)
    outputs_hidden = Dropout(dropout_rate)(dense_hidden)
    dense_output = TimeDistributed(Dense(1, activation='linear'), name='output_dense_final')(outputs_hidden)

    model = Model([encoder_inputs, decoder_inputs], dense_output)
    # 降低初始学习率至0.0005，避免训练震荡
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# 训练过程实时打印回调
class TrainingMetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"训练集 - 损失: {logs.get('loss'):.4f}, MAE: {logs.get('mae'):.4f}")
        print(f"验证集 - 损失: {logs.get('val_loss'):.4f}, MAE: {logs.get('val_mae'):.4f}")

def plot_history(history, save_path):
    """增强版：绘制更详细的训练历史并保存。"""
    plt.figure(figsize=(16, 12))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', marker='o', markersize=3)
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='s', markersize=3)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # MAE曲线
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', marker='o', markersize=3)
    plt.plot(history.history['val_mae'], label='Validation MAE', marker='s', markersize=3)
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 损失曲线（放大后期）
    plt.subplot(2, 2, 3)
    start_epoch = max(0, len(history.history['loss']) - 20)  # 只显示最后20轮
    plt.plot(range(start_epoch, len(history.history['loss'])), 
             history.history['loss'][start_epoch:], 
             label='Training Loss', marker='o', markersize=3)
    plt.plot(range(start_epoch, len(history.history['val_loss'])), 
             history.history['val_loss'][start_epoch:], 
             label='Validation Loss', marker='s', markersize=3)
    plt.title('Recent Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # MAE曲线（放大后期）
    plt.subplot(2, 2, 4)
    plt.plot(range(start_epoch, len(history.history['mae'])), 
             history.history['mae'][start_epoch:], 
             label='Training MAE', marker='o', markersize=3)
    plt.plot(range(start_epoch, len(history.history['val_mae'])), 
             history.history['val_mae'][start_epoch:], 
             label='Validation MAE', marker='s', markersize=3)
    plt.title('Recent Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"增强版训练历史图表已保存至 {save_path}")

def plot_predictions_vs_actual(predictions, true_values, timestamps, title, save_path, sample_size=100):
    """绘制预测值与实际值的对比图"""
    n_samples, n_timesteps, _ = predictions.shape
    
    if n_samples > sample_size:
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    else:
        sample_indices = range(n_samples)
    
    plt.figure(figsize=(15, 8))
    
    # 绘制多个样本的预测与实际值
    for i, idx in enumerate(sample_indices[:5]):
        plt.subplot(2, 3, i+1)
        plt.plot(pd.to_datetime(timestamps[idx]), true_values[idx, :, 0], label='Actual', alpha=0.7)
        plt.plot(pd.to_datetime(timestamps[idx]), predictions[idx, :, 0], label='Predicted', linestyle='--')
        plt.title(f'Sample {idx}')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.legend()
    
    # 绘制所有样本的误差分布
    plt.subplot(2, 3, 6)
    all_errors = np.abs(predictions - true_values).flatten()
    plt.hist(all_errors, bins=30, alpha=0.7)
    plt.axvline(np.mean(all_errors), color='r', linestyle='dashed', linewidth=1, label=f'Mean Error: {np.mean(all_errors):.2f}')
    plt.title('Error Distribution')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"预测对比图表已保存至 {save_path}")

def plot_aggregated_results(df, title, save_path):
    """绘制聚合后的预测结果时间序列图"""
    plt.figure(figsize=(18, 6))
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    plt.plot(df['timestamp'], df['actual_flow'], label='Actual Flow', alpha=0.7)
    plt.plot(df['timestamp'], df['predicted_flow'], label='Predicted Flow', linestyle='--', alpha=0.8)
    
    plt.fill_between(df['timestamp'], 
                    df['actual_flow'], 
                    df['predicted_flow'], 
                    color='gray', alpha=0.2, label='Error')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Flow Value', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    
    every_nth = max(1, len(df) // 10)
    plt.xticks(df['timestamp'][::every_nth], rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"聚合结果时间序列图已保存至 {save_path}")

def save_hourly_predictions(predictions, true_values, target_timestamps, scaler, save_path):
    """使用精确的时间戳数组保存小时级预测"""
    assert predictions.shape[0:2] == true_values.shape[0:2] == target_timestamps.shape[0:2], "Shape mismatch!"

    n_samples, n_timesteps, _ = predictions.shape
    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(n_samples, n_timesteps, 1)
    true_values_inv = scaler.inverse_transform(true_values.reshape(-1, 1)).reshape(n_samples, n_timesteps, 1)

    results = []
    for i in range(n_samples):
        for j in range(n_timesteps):
            results.append({
                'timestamp': pd.to_datetime(target_timestamps[i, j]).strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_flow': predictions_inv[i, j, 0],
                'actual_flow': true_values_inv[i, j, 0]
            })

    results_df = pd.DataFrame(results).sort_values('timestamp')
    results_df.to_csv(save_path, index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f"小时级预测结果已保存至: {save_path}")
    return predictions_inv, true_values_inv

def aggregate_hourly_predictions(input_csv_path, output_csv_path):
    """聚合小时级预测结果"""
    print(f"\n聚合预测结果: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"警告: 文件未找到，跳过聚合: {input_csv_path}")
        return None

    aggregated_df = df.groupby('timestamp').agg({
        'actual_flow': 'first',
        'predicted_flow': ['mean', 'size']
    })
    aggregated_df.columns = ['actual_flow', 'predicted_flow_mean', 'prediction_count']
    aggregated_df = aggregated_df.reset_index()

    aggregated_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f"聚合后的预测已保存至: {output_csv_path}")
    return aggregated_df


# ==============================================================================
# 2. 主执行程序 (优化训练策略)
# ==============================================================================
if __name__ == '__main__':
    # --- 步骤 1: 基础配置 ---
    DATA_PATH = r'C:\Users\24766\Desktop\final_work\final_demo\final_demo\final_demo\INTERFACE_FINAL\preparation_in\processed_zggg_data_optimized_new.npz'
    SCALER_PATH = r'C:\Users\24766\Desktop\final_work\final_demo\final_demo\final_demo\INTERFACE_FINAL\preparation_in\zggg_optimized_scalers_new.gz'
    MODEL_SAVE_DIR = r'C:\Users\24766\Desktop\final_work\final_demo\final_demo\final_demo\INTERFACE_FINAL\model_output_dual_corrected'
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # --- 步骤 2: 定义时段判断函数 ---
    def is_busy(hour):
        return 7 <= hour <= 22

    training_configs = [
        {"name": "busy", "period_checker": is_busy, "target_seq_len": 16},
        {"name": "non_busy", "period_checker": lambda h: not is_busy(h), "target_seq_len": 8}
    ]

    # --- 步骤 3: 加载完整数据 ---
    print("="*60)
    print("加载完整数据集...")
    with np.load(DATA_PATH, allow_pickle=True) as data:
        X_train_encoder = data['X_train_encoder']
        X_train_decoder_full = data['X_train_decoder']
        y_train_full = data['y_train']
        train_timestamps_full = data['train_timestamps']

        X_val_encoder = data['X_val_encoder']
        X_val_decoder_full = data['X_val_decoder']
        y_val_full = data['y_val']
        val_timestamps_full = data['val_timestamps']

    print("加载归一化器...")
    with gzip.open(SCALER_PATH, 'rb') as f:
        scaler = load(f)
    target_scaler = scaler['target']
    print("数据和归一化器加载完毕。")

    aggregated_results = {}

    # --- 步骤 4: 循环训练每个模型 ---
    for config in training_configs:
        period_name = config["name"]
        period_checker = config["period_checker"]
        target_seq_len = config["target_seq_len"]

        print("\n" + "="*60)
        print(f"开始处理【{period_name.upper()}】时段模型")
        print("="*60)

        # 4a. 动态切片数据
        print(f"正在为【{period_name}】时段动态准备数据...")

        # 训练集动态切片
        train_indices_list = [
            [j for j, ts in enumerate(sample_ts) if period_checker(pd.to_datetime(ts).hour)]
            for sample_ts in tqdm(train_timestamps_full, desc=f"Scanning Train Timestamps for '{period_name}'")
        ]
        assert all(len(indices) == target_seq_len for indices in train_indices_list), f"Train data error: Not all samples have {target_seq_len} '{period_name}' hours!"

        y_train_period = np.array([y_train_full[i, indices, :] for i, indices in enumerate(train_indices_list)])
        X_train_decoder_period = np.array([X_train_decoder_full[i, indices, :] for i, indices in enumerate(train_indices_list)])
        train_timestamps_period = np.array([train_timestamps_full[i, indices] for i, indices in enumerate(train_indices_list)])

        # 验证集动态切片
        val_indices_list = [
            [j for j, ts in enumerate(sample_ts) if period_checker(pd.to_datetime(ts).hour)]
            for sample_ts in tqdm(val_timestamps_full, desc=f"Scanning Val Timestamps for '{period_name}'")
        ]
        assert all(len(indices) == target_seq_len for indices in val_indices_list), f"Validation data error: Not all samples have {target_seq_len} '{period_name}' hours!"

        y_val_period = np.array([y_val_full[i, indices, :] for i, indices in enumerate(val_indices_list)])
        X_val_decoder_period = np.array([X_val_decoder_full[i, indices, :] for i, indices in enumerate(val_indices_list)])
        val_timestamps_period = np.array([val_timestamps_full[i, indices] for i, indices in enumerate(val_indices_list)])

        print(f"动态切片完成。训练目标y形状: {y_train_period.shape}, 验证目标y形状: {y_val_period.shape}")

        # 4b. 构建模型（使用增强正则化版本）
        model = build_seq2seq_attention_model(
            input_seq_len=X_train_encoder.shape[1],
            target_seq_len=target_seq_len,
            num_encoder_features=X_train_encoder.shape[2],
            num_decoder_features=X_train_decoder_period.shape[2]
        )

        # 4c. 训练模型（添加学习率衰减+更严格早停）
        best_model_path = os.path.join(MODEL_SAVE_DIR, f'best_model_{period_name}.h5')
        checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        # 早停patience从10→8，更早终止无效训练
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True, verbose=1)
        # 新增学习率衰减：验证损失5轮不下降则学习率减半
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5,    # 衰减系数
            patience=5,    # 等待轮数
            min_lr=1e-6,   # 最小学习率
            verbose=1
        )
        metrics_callback = TrainingMetricsCallback()

        print(f"\n开始训练【{period_name}】模型...")
        history = model.fit(
            [X_train_encoder, X_train_decoder_period], y_train_period,
            epochs=100, batch_size=32,
            validation_data=([X_val_encoder, X_val_decoder_period], y_val_period),
            callbacks=[checkpoint, early_stopping, reduce_lr, metrics_callback]  # 整合新回调
        )

        # 4d. 保存和评估
        print(f"【{period_name}】模型训练完成。")
        plot_history(history, os.path.join(MODEL_SAVE_DIR, f'training_history_{period_name}_new.png'))

        best_model = load_model(best_model_path)
        val_predictions = best_model.predict([X_val_encoder, X_val_decoder_period])

        val_hourly_path = os.path.join(MODEL_SAVE_DIR, f'validation_hourly_predictions_{period_name}_new.csv')
        pred_inv, true_inv = save_hourly_predictions(val_predictions, y_val_period, val_timestamps_period, target_scaler, val_hourly_path)
        
        plot_path = os.path.join(MODEL_SAVE_DIR, f'predictions_vs_actual_{period_name}_new.png')
        plot_predictions_vs_actual(pred_inv, true_inv, val_timestamps_period, 
                                  f'Predictions vs Actual Values ({period_name} period)', 
                                  plot_path)

        val_agg_path = os.path.join(MODEL_SAVE_DIR, f'validation_aggregated_predictions_{period_name}_new.csv')
        agg_df = aggregate_hourly_predictions(val_hourly_path, val_agg_path)
        aggregated_results[period_name] = agg_df

        if agg_df is not None:
            agg_plot_path = os.path.join(MODEL_SAVE_DIR, f'aggregated_predictions_{period_name}_new.png')
            plot_aggregated_results(agg_df.rename(columns={'predicted_flow_mean': 'predicted_flow'}),
                                   f'Aggregated Predictions ({period_name} period)',
                                   agg_plot_path)


    # ==============================================================================
    # 5. 合并两个模型的预测结果，并择优选择
    # ==============================================================================
    print("\n" + "="*60)
    print("择优合并所有时段的预测结果...")
    try:
        # 加载两个模型聚合后的预测结果
        busy_pred_path = os.path.join(MODEL_SAVE_DIR, 'validation_aggregated_predictions_busy_new.csv')
        non_busy_pred_path = os.path.join(MODEL_SAVE_DIR, 'validation_aggregated_predictions_non_busy_new.csv')

        df_busy = pd.read_csv(busy_pred_path)
        df_non_busy = pd.read_csv(non_busy_pred_path)

        df_busy.rename(columns={'predicted_flow_mean': 'predicted_flow_busy'}, inplace=True)
        df_non_busy.rename(columns={'predicted_flow_mean': 'predicted_flow_non_busy'}, inplace=True)

        # 外连接合并
        df_merged = pd.merge(
            df_busy[['timestamp', 'actual_flow', 'predicted_flow_busy']],
            df_non_busy[['timestamp', 'actual_flow', 'predicted_flow_non_busy']],
            on=['timestamp', 'actual_flow'],
            how='outer'
        )

        # 计算误差并择优
        df_merged['error_busy'] = (df_merged['predicted_flow_busy'] - df_merged['actual_flow']).abs()
        df_merged['error_non_busy'] = (df_merged['predicted_flow_non_busy'] - df_merged['actual_flow']).abs()

        df_merged['final_prediction'] = np.where(
            df_merged['error_busy'].fillna(np.inf) < df_merged['error_non_busy'].fillna(np.inf),
            df_merged['predicted_flow_busy'],
            df_merged['predicted_flow_non_busy']
        )
        
        df_final = df_merged[['timestamp', 'actual_flow', 'final_prediction']].rename(columns={'final_prediction': 'predicted_flow'})
        df_final = df_final.sort_values('timestamp')
        combined_path = os.path.join(MODEL_SAVE_DIR, 'validation_FINAL_optimal_predictions_new.csv')
        df_final.to_csv(combined_path, index=False, encoding='utf-8-sig', float_format='%.2f')
        print(f"成功择优合并！最终优化预测文件已保存至: {combined_path}")

        # 可视化最终结果
        final_plot_path = os.path.join(MODEL_SAVE_DIR, 'final_merged_predictions_new.png')
        plot_aggregated_results(df_final, 'Final Optimized Predictions (Combined)', final_plot_path)

        # 误差对比
        plt.figure(figsize=(12, 6))
        df_merged['final_error'] = (df_merged['final_prediction'] - df_merged['actual_flow']).abs()
        
        avg_errors = {
            'Busy Model': df_merged['error_busy'].mean(skipna=True),
            'Non-Busy Model': df_merged['error_non_busy'].mean(skipna=True),
            'Optimal Combined': df_merged['final_error'].mean(skipna=True)
        }
        
        plt.bar(avg_errors.keys(), avg_errors.values(), color=['blue', 'orange', 'green'])
        for i, v in enumerate(avg_errors.values()):
            plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
        plt.title('Average Absolute Error Comparison')
        plt.ylabel('Average Absolute Error')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        error_compare_path = os.path.join(MODEL_SAVE_DIR, 'error_comparison_new.png')
        plt.savefig(error_compare_path, dpi=300)
        plt.close()
        print(f"模型误差对比图已保存至: {error_compare_path}")

    except FileNotFoundError as e:
        print(f"警告: 文件未找到: {e.filename}。跳过最终合并步骤。")
    except Exception as e:
        print(f"合并过程中发生未知错误: {e}")

    print("\n--- 所有流程执行完毕 ---")