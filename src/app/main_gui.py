import pandas as pd
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import sys
import os
from datetime import datetime
import shutil
# 新增：导入可视化相关库
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates

try:
    from run_prediction_in import predict_future_flow as predict_in_flow
    from run_prediction_out import predict_future_flow as predict_out_flow
    from flight_filter import filter_departure_flights
    from status_simulation import run_simulation
    from optimization_model import run_optimization
except ImportError as e:
    messagebox.showerror("依赖错误", f"无法导入模块: {e.name}。\n请确保所有 .py 文件都在正确的路径下。")
    sys.exit()

# 新增：设置matplotlib字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class TextRedirector:
    def __init__(self, widget): self.widget = widget
    def write(self, s): self.widget.after(0, self.update_text, s)
    def update_text(self, s): self.widget.insert(tk.END, s); self.widget.see(tk.END)
    def flush(self): pass

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("机场流量预测、推演与优化工具")
        self.master.geometry("1000x900")  # 扩大窗口以容纳图表
        self.pack(fill="both", expand=True, padx=10, pady=10)
        self.shared_data = {}
        self.create_widgets()
        self.original_stdout = sys.stdout
        sys.stdout = TextRedirector(self.log_text)

    def create_widgets(self):
        # --- 原有步骤1-6代码保持不变 ---
        # 步骤 1：选择文件夹
        step1_frame = tk.LabelFrame(self, text="步骤 1：选择文件夹", padx=10, pady=10); step1_frame.pack(fill="x", pady=5)
        work_dir_frame = tk.Frame(step1_frame); work_dir_frame.pack(fill='x', pady=2)
        tk.Label(work_dir_frame, text="工作/模型文件夹:", width=18, anchor='w').pack(side="left")
        self.work_dir_var = tk.StringVar(); tk.Entry(work_dir_frame, textvariable=self.work_dir_var, state='readonly').pack(side="left", fill="x", expand=True, padx=5)
        self.select_work_dir_button = tk.Button(work_dir_frame, text="选择...", command=self.select_work_dir); self.select_work_dir_button.pack(side="left")
        tk.Label(step1_frame, text="此文件夹需包含模型(.h5, .gz)和备用的平均数据(.csv)", justify=tk.LEFT, fg="gray").pack(pady=(0, 5), anchor='w')

        # 步骤 2: 提供天气预报
        step2_frame = tk.LabelFrame(self, text="步骤 2：提供天气预报", padx=10, pady=10); step2_frame.pack(fill="x", pady=5)
        forecast_frame = tk.Frame(step2_frame); forecast_frame.pack(fill='x', pady=2)
        tk.Label(forecast_frame, text="未来24小时天气文件:", width=18, anchor='w').pack(side="left")
        self.forecast_file_path_var = tk.StringVar(); tk.Entry(forecast_frame, textvariable=self.forecast_file_path_var, state='readonly').pack(side="left", fill="x", expand=True, padx=5)
        self.select_forecast_file_button = tk.Button(forecast_frame, text="选择...", command=self.select_forecast_file); self.select_forecast_file_button.pack(side="left")

        # 步骤 3: 选择预测类型
        step3_frame = tk.LabelFrame(self, text="步骤 3：选择预测类型", padx=10, pady=10); step3_frame.pack(fill="x", pady=5)
        self.prediction_type_var = tk.StringVar(value="in"); radio_frame = tk.Frame(step3_frame); radio_frame.pack(pady=5)
        self.radio_in = ttk.Radiobutton(radio_frame, text="预测进港流量", variable=self.prediction_type_var, value="in", command=self.on_prediction_type_change); self.radio_in.pack(side="left", padx=20)
        self.radio_out = ttk.Radiobutton(radio_frame, text="预测出港流量", variable=self.prediction_type_var, value="out", command=self.on_prediction_type_change); self.radio_out.pack(side="left", padx=20)

        # 步骤 4: 执行流量预测
        step4_frame = tk.LabelFrame(self, text="步骤 4：执行流量预测", padx=10, pady=10); step4_frame.pack(fill="x", pady=5)
        self.predict_button = tk.Button(step4_frame, text="4. 生成未来24小时流量预测", command=lambda: self.start_thread(self.run_prediction), font=("Arial", 12, "bold"), bg="#F5B7B1", state="disabled")
        self.predict_button.pack(pady=5, fill='x')

        # 步骤 5: 航班状态推演
        self.step5_frame = tk.LabelFrame(self, text="步骤 5：航班状态推演 (仅出港)", padx=10, pady=10); self.step5_frame.pack(fill="x", pady=5)
        flight_plan_frame = tk.Frame(self.step5_frame); flight_plan_frame.pack(fill='x', pady=2)
        tk.Label(flight_plan_frame, text="航班计划总文件:", width=18, anchor='w').pack(side="left")
        self.flight_plan_path_var = tk.StringVar()
        self.flight_plan_entry = tk.Entry(flight_plan_frame, textvariable=self.flight_plan_path_var, state='readonly'); self.flight_plan_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.select_flight_plan_button = tk.Button(flight_plan_frame, text="选择...", command=self.select_flight_plan); self.select_flight_plan_button.pack(side="left")
        self.simulation_button = tk.Button(self.step5_frame, text="5. 基于预测结果推演航班运行状态", command=lambda: self.start_thread(self.run_simulation_wrapper), font=("Arial", 12, "bold"), bg="#AED6F1", state="disabled"); self.simulation_button.pack(pady=5, fill='x')

        # 步骤 6: 协同恢复优化
        self.step6_frame = tk.LabelFrame(self, text="步骤 6：协同恢复优化 (仅出港)", padx=10, pady=10); self.step6_frame.pack(fill="x", pady=5)
        self.optimize_button = tk.Button(self.step6_frame, text="6. 基于推演结果进行航班协同恢复优化", command=lambda: self.start_thread(self.run_optimization_wrapper), font=("Arial", 12, "bold"), bg="#D5F5E3", state="disabled"); self.optimize_button.pack(pady=5, fill='x')

        # --- 新增：可视化区域（标签页）---
        visual_frame = tk.LabelFrame(self, text="结果可视化", padx=10, pady=10); visual_frame.pack(fill="both", expand=True, pady=5)
        self.visual_notebook = ttk.Notebook(visual_frame)
        self.visual_notebook.pack(fill="both", expand=True)

        # 1. 预测结果可视化标签页
        self.pred_visual_tab = tk.Frame(self.visual_notebook)
        self.visual_notebook.add(self.pred_visual_tab, text="4. 流量预测趋势")
        # 2. 推演结果可视化标签页
        self.sim_visual_tab = tk.Frame(self.visual_notebook)
        self.visual_notebook.add(self.sim_visual_tab, text="5. 航班状态分布")
        # 3. 优化结果可视化标签页
        self.opt_visual_tab = tk.Frame(self.visual_notebook)
        self.visual_notebook.add(self.opt_visual_tab, text="6. 优化前后对比")

        # --- 原有日志区域 ---
        log_frame = tk.LabelFrame(self, text="处理日志", padx=10, pady=10); log_frame.pack(fill="both", expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD); self.log_text.pack(fill="both", expand=True)

        # 原有按钮列表
        self.all_buttons = [self.select_work_dir_button, self.select_forecast_file_button, self.select_flight_plan_button, self.predict_button, self.simulation_button, self.radio_in, self.radio_out, self.optimize_button]
        
        self.on_prediction_type_change()

    # --- 原有选择文件夹/文件方法保持不变 ---
    def select_work_dir(self): path = filedialog.askdirectory(); self.work_dir_var.set(path); self.check_button_states()
    def select_forecast_file(self): path = filedialog.askopenfilename(filetypes=(("CSV", "*.csv"),)); self.forecast_file_path_var.set(path); self.check_button_states()
    def select_flight_plan(self): path = filedialog.askopenfilename(filetypes=(("Excel", "*.xlsx"),("CSV", "*.csv"),)); self.flight_plan_path_var.set(path); self.check_button_states()

    # --- 原有状态切换/按钮检查方法保持不变 ---
    def on_prediction_type_change(self):
        self.shared_data = {}
        self.flight_plan_path_var.set("")
        self.check_button_states()
        # 新增：进港模式下隐藏推演/优化可视化标签
        if self.prediction_type_var.get() == "in":
            self.visual_notebook.tab(self.sim_visual_tab, state="disabled")
            self.visual_notebook.tab(self.opt_visual_tab, state="disabled")
        else:
            self.visual_notebook.tab(self.sim_visual_tab, state="normal")
            self.visual_notebook.tab(self.opt_visual_tab, state="normal")

    def check_button_states(self):
        self.select_work_dir_button.config(state='normal')
        self.select_forecast_file_button.config(state='normal')

        can_predict = self.work_dir_var.get() and self.forecast_file_path_var.get()
        self.predict_button.config(state="normal" if can_predict else "disabled")
        step5_widgets = [self.flight_plan_entry, self.select_flight_plan_button, self.simulation_button]
        step6_widgets = [self.optimize_button]
    
        if self.prediction_type_var.get() == "out":
            self.select_flight_plan_button.config(state='normal')
            self.flight_plan_entry.config(state='readonly')

            can_simulate = self.shared_data.get("prediction_success") and self.flight_plan_path_var.get()
            self.simulation_button.config(state="normal" if can_simulate else "disabled")

            can_optimize = self.shared_data.get("simulation_success")
            self.optimize_button.config(state="normal" if can_optimize else "disabled")
            
        else:
            for widget in step5_widgets: widget.config(state='disabled')
            for widget in step6_widgets: widget.config(state='disabled')

    # --- 原有线程/按钮状态控制方法保持不变 ---
    def set_buttons_state(self, state):
        if state == 'disabled':
            for btn in self.all_buttons:
                btn.config(state='disabled')
        elif state == 'normal':
            self.check_button_states()

    def start_thread(self, target_func): self.set_buttons_state("disabled"); threading.Thread(target=target_func, daemon=True).start()

    # --- 新增：可视化核心方法 ---
    def clear_visual_tab(self, tab_frame):
        """清空标签页中的原有图表"""
        for widget in tab_frame.winfo_children():
            widget.destroy()

    def plot_prediction_result(self, result_path, prediction_type):
        """绘制流量预测结果（折线图：24小时流量趋势）"""
        self.clear_visual_tab(self.pred_visual_tab)
        try:
            # 读取预测结果
            df = pd.read_csv(result_path)
            
            # 关键修改：尝试识别常见的时间列名（根据实际文件调整）
            time_columns = ['datetime', 'timestamp', 'time', 'date_time']  # 常见时间列名列表
            time_col = None
            for col in time_columns:
                if col in df.columns:
                    time_col = col
                    break
            
            # 如果找不到时间列，抛出明确错误
            if time_col is None:
                raise ValueError(f"预测结果文件中未找到时间列！\n请检查文件是否包含以下列之一：{time_columns}")
            
            # 解析时间列
            df[time_col] = pd.to_datetime(df[time_col])
            
            # 关键修改：确认流量数据列名（根据实际文件调整，常见如'flow'、'predicted_flow'等）
            flow_columns = ['predicted_flow', 'flow', 'traffic']  # 常见流量列名列表
            flow_col = None
            for col in flow_columns:
                if col in df.columns:
                    flow_col = col
                    break
            
            if flow_col is None:
                raise ValueError(f"预测结果文件中未找到流量数据列！\n请检查文件是否包含以下列之一：{flow_columns}")
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
            ax.plot(df[time_col], df[flow_col],  # 使用识别到的列名
                    color='#E74C3C' if prediction_type == 'in' else '#3498DB',
                    linewidth=2.5, marker='o', markersize=4, label='预测流量')
            
            # 设置图表格式
            ax.set_title(f'未来24小时{"进港" if prediction_type == "in" else "出港"}流量预测趋势', fontsize=14, pad=20)
            ax.set_xlabel('时间', fontsize=12)
            ax.set_ylabel('航班流量（架次/小时）', fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 小时:分钟格式
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))  # 每3小时一个刻度
            plt.xticks(rotation=45)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()

            # 嵌入Tkinter窗口
            canvas = FigureCanvasTkAgg(fig, master=self.pred_visual_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # 添加工具栏
            toolbar = NavigationToolbar2Tk(canvas, self.pred_visual_tab)
            toolbar.update()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            print(f"绘制预测图表失败: {e}")
            # 显示更友好的错误提示，指导用户检查文件
            tk.Label(
                self.pred_visual_tab, 
                text=f"图表加载失败：{str(e)}\n请检查CSV文件的列名是否正确", 
                fg="red",
                wraplength=400  # 自动换行
            ).pack(pady=20)

    def plot_simulation_result(self, result_path):
        """绘制航班推演结果（柱状图：航班状态分布）"""
        self.clear_visual_tab(self.sim_visual_tab)
        try:
            # 读取推演结果（假设文件包含'flight_status'列）
            df = pd.read_csv(result_path)
            status_count = df['flight_status'].value_counts()  # 统计各状态航班数量（如：正常、延误、取消）
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
            colors = ['#2ECC71', '#F39C12', '#E74C3C', '#9B59B6']  # 正常、轻微延误、严重延误、取消
            bars = ax.bar(status_count.index, status_count.values, color=colors[:len(status_count)])
            
            # 在柱子上添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=11)
            
            # 设置图表格式
            ax.set_title('航班运行状态推演结果分布', fontsize=14, pad=20)
            ax.set_xlabel('航班状态', fontsize=12)
            ax.set_ylabel('航班数量（架次）', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()

            # 嵌入Tkinter窗口
            canvas = FigureCanvasTkAgg(fig, master=self.sim_visual_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            toolbar = NavigationToolbar2Tk(canvas, self.sim_visual_tab)
            toolbar.update()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            print(f"绘制推演图表失败: {e}")
            tk.Label(self.sim_visual_tab, text=f"图表加载失败：{str(e)}", fg="red").pack(pady=20)

    def plot_optimization_result(self, opt_result_path, sim_result_path):
        """绘制优化结果（双柱状图：优化前后延误对比）"""
        self.clear_visual_tab(self.opt_visual_tab)
        try:
            # 读取优化前后数据（假设包含'flight_no'和'delay_time'列）
            sim_df = pd.read_csv(sim_result_path)
            opt_df = pd.read_csv(opt_result_path)
            
            # 计算平均延误时间（仅统计延误航班）
            sim_delay = sim_df[sim_df['delay_time'] > 0]['delay_time'].mean()
            opt_delay = opt_df[opt_df['delay_time'] > 0]['delay_time'].mean()
            delay_data = [sim_delay, opt_delay]
            labels = ['优化前', '优化后']

            # 创建图表
            fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
            bars = ax.bar(labels, delay_data, color=['#F39C12', '#2ECC71'], width=0.5)
            
            # 添加数值标签
            for bar, val in zip(bars, delay_data):
                ax.text(bar.get_x() + bar.get_width()/2., val + 0.5,
                        f'{val:.1f}分钟', ha='center', va='bottom', fontsize=11)
            
            # 设置图表格式
            ax.set_title('航班协同恢复优化效果对比', fontsize=14, pad=20)
            ax.set_ylabel('平均延误时间（分钟）', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()

            # 嵌入Tkinter窗口
            canvas = FigureCanvasTkAgg(fig, master=self.opt_visual_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            toolbar = NavigationToolbar2Tk(canvas, self.opt_visual_tab)
            toolbar.update()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            print(f"绘制优化图表失败: {e}")
            tk.Label(self.opt_visual_tab, text=f"图表加载失败：{str(e)}", fg="red").pack(pady=20)

    # --- 原有核心业务方法（新增可视化调用）---
    def run_prediction(self):
        self.predict_button.config(text="正在预测...")
        self.shared_data = {}
        
        work_dir_base = self.work_dir_var.get()
        forecast_file = self.forecast_file_path_var.get()
        prediction_type = self.prediction_type_var.get()

        if not all([work_dir_base, forecast_file]): 
            messagebox.showwarning("输入错误", "请确保已选择工作/模型文件夹和未来天气预报文件！")
            self.set_buttons_state("normal")
            return
        
        try:
            if prediction_type == "in": 
                model_subfolder, predict_function = "preparation_in", predict_in_flow
                print("\n" + "="*20 + " 开始预测【进港】流量 " + "="*20)
            else: 
                model_subfolder, predict_function = "preparation_out", predict_out_flow
                print("\n" + "="*20 + " 开始预测【出港】流量 " + "="*20)
            
            model_dir = os.path.join(work_dir_base, model_subfolder)
            if not os.path.isdir(model_dir): 
                messagebox.showerror("路径错误", f"找不到指定的模型子文件夹: {model_dir}")
                self.set_buttons_state("normal")
                return
            
            forecast_df = pd.read_csv(forecast_file)
            if 'report_day' in forecast_df.columns: 
                prediction_date_str = f"{datetime.now().year}-{forecast_df['report_day'].iloc[0]}"
            elif 'timestamp' in forecast_df.columns: 
                prediction_date_str = pd.to_datetime(forecast_df['timestamp'].iloc[0]).strftime('%Y-%m-%d')
            else: 
                raise ValueError("天气预报文件中找不到'report_day'或'timestamp'列。")
            print(f"自动确定预测日期为: {prediction_date_str}")
            
            target_forecast_path = os.path.join(model_dir, 'future_weather_forecast.csv')
            shutil.copy(forecast_file, target_forecast_path)
            print(f"已将天气预报文件复制到模型目录: {target_forecast_path}")
            
            success = predict_function(model_dir=model_dir, work_dir=model_dir, prediction_start_date_str=prediction_date_str)
            
            if success:
                self.shared_data["prediction_success"] = True
                self.shared_data["prediction_date"] = prediction_date_str
                self.shared_data["run_dir"] = model_dir
                output_file = os.path.join(model_dir, 'future_24h_prediction.csv')
                messagebox.showinfo("成功", f"流量预测完成！\n结果已保存至:\n{output_file}")
                # 新增：调用预测可视化
                self.plot_prediction_result(output_file, prediction_type)
            else:
                messagebox.showerror("失败", "流量预测过程中发生错误，请查看日志。")
        except Exception as e:
            print(f"执行预测时发生严重错误: {e}")
            messagebox.showerror("严重错误", f"执行预测时发生错误: \n{e}")
        finally:
            self.set_buttons_state("normal")
            self.predict_button.config(text="4. 生成未来24小时流量预测")

    def run_simulation_wrapper(self):
        self.simulation_button.config(text="正在推演...")
        run_dir, prediction_date, flight_plan = self.shared_data.get("run_dir"), self.shared_data.get("prediction_date"), self.flight_plan_path_var.get()
        if not all([run_dir, prediction_date, flight_plan]): 
            messagebox.showerror("错误", "无法开始推演，缺少前一阶段信息。")
            self.set_buttons_state("normal")
            return

        try:
            print("\n" + "="*20 + " 阶段一 (推演前置)：筛选当日航班计划 " + "="*20)
            daily_flight_plan = os.path.join(run_dir, f'zggg_departures_only_{prediction_date}.csv')
            filter_departure_flights(input_filepath=flight_plan, output_filepath=daily_flight_plan, target_airport_code='ZGGG', target_date_str=prediction_date)
            
            print("\n" + "="*20 + " 阶段二 (核心推演)：运行状态仿真 " + "="*20)
            capacity_forecast = os.path.join(run_dir, 'future_24h_prediction.csv')
            if not os.path.exists(capacity_forecast): 
                messagebox.showerror("文件缺失", f"找不到流量预测结果文件:\n{capacity_forecast}")
                self.set_buttons_state("normal")
                return

            run_simulation(flights_filepath=daily_flight_plan, capacity_filepath=capacity_forecast, target_airport='ZGGG', sim_start_time=datetime.strptime(prediction_date, '%Y-%m-%d'), output_dir=run_dir)
            self.shared_data["simulation_success"] = True
            sim_result_file = os.path.join(run_dir, "full_flight_log_ZGGG.csv")  # 推演结果文件
            messagebox.showinfo("成功", f"状态推演完成！\n分析报告和日志已保存到:\n{run_dir}")
            # 新增：调用推演可视化
            self.plot_simulation_result(sim_result_file)

        except Exception as e:
            self.shared_data["simulation_success"] = False
            print(f"执行状态推演时发生严重错误: {e}")
            messagebox.showerror("严重错误", f"执行状态推演时发生错误: \n{e}")
        finally:
            self.set_buttons_state("normal")
            self.simulation_button.config(text="5. 基于预测结果推演航班运行状态")

    def run_optimization_wrapper(self):
        self.optimize_button.config(text="正在优化...")
        run_dir = self.shared_data.get("run_dir")
        if not run_dir: 
            messagebox.showerror("错误", "无法开始优化，缺少前一阶段的目录信息。")
            self.set_buttons_state("normal")
            return
        try:
            sim_log = os.path.join(run_dir, "full_flight_log_ZGGG.csv")
            if not os.path.exists(sim_log): 
                messagebox.showerror("文件缺失", f"找不到仿真日志文件:\n{sim_log}")
                self.set_buttons_state("normal")
                return
            
            success = run_optimization(input_filepath=sim_log, output_dir=run_dir)
            opt_result_file = os.path.join(run_dir, "optimal_plan.csv")  # 优化结果文件
            
            if success:
                messagebox.showinfo("成功", f"协同恢复优化完成！\n优化后的计划已保存至:\n{opt_result_file}")
                # 新增：调用优化可视化（对比优化前后）
                self.plot_optimization_result(opt_result_file, sim_log)
            else:
                messagebox.showerror("失败", "优化过程中发生错误，请查看日志。")
        except Exception as e:
            print(f"执行优化时发生严重错误: {e}")
            messagebox.showerror("严重错误", f"执行优化时发生错误: \n{e}")
        finally:
            self.set_buttons_state("normal")
            self.optimize_button.config(text="6. 基于推演结果进行航班协同恢复优化")

    def on_closing(self): 
        sys.stdout = self.original_stdout
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use('vista')
    app = Application(master=root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()