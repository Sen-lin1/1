import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 引入字体管理模块

# ==========================================
# 1. 基础配置与路径修复
# ==========================================
try:
    # 强制将工作目录切换到当前脚本所在的文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except:
    pass

# ==========================================
# 2. 彻底解决中文乱码 (双保险逻辑)
# ==========================================
def set_chinese_font():
    """
    自动寻找可用的中文字体。
    优先寻找当前目录下的 SimHei.ttf，其次寻找系统字体。
    """
    # 方案 A: 优先使用随项目上传的字体文件 (最稳妥，推荐！)
    local_font_path = 'SimHei.ttf'  # 请确保您把这个文件上传到了 GitHub
    if os.path.exists(local_font_path):
        # 注册字体
        fm.fontManager.addfont(local_font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # st.toast("已加载本地 SimHei 字体", icon="✅") # 调试用
    else:
        # 方案 B: 如果没上传文件，尝试系统自带的常见中文字体
        system_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
        plt.rcParams['font.sans-serif'] = system_fonts
    
    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

# 执行字体设置
set_chinese_font()

# ==========================================
# 3. 核心类定义：EFTM 模型
# ==========================================
class EFTMModel:
    def __init__(self, w_cb=0.385412, w_xgb=0.294103, w_lgbm=0.211438, w_ab=0.109047):
        self.w_cb = w_cb
        self.w_xgb = w_xgb
        self.w_lgbm = w_lgbm
        self.w_ab = w_ab

    def predict(self, pred_cb, pred_xgb, pred_lgbm, pred_ab):
        p_cb = np.array(pred_cb)
        p_xgb = np.array(pred_xgb)
        p_lgbm = np.array(pred_lgbm)
        p_ab = np.array(pred_ab)
        return (self.w_cb * p_cb) + (self.w_xgb * p_xgb) + (self.w_lgbm * p_lgbm) + (self.w_ab * p_ab)

# ==========================================
# 4. 工具函数
# ==========================================
@st.cache_resource
def load_models():
    """加载已训练好的模型文件"""
    required_files = ['model_cb.pkl', 'model_xgb.pkl', 'model_lgbm.pkl', 'model_ab.pkl', 'feature_names.pkl']

    # 检查文件是否存在
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        return None, None, None, None, None, missing

    # 加载模型
    try:
        cb = joblib.load('model_cb.pkl')
        xgb_m = joblib.load('model_xgb.pkl')
        lgbm = joblib.load('model_lgbm.pkl')
        ab = joblib.load('model_ab.pkl')
        feats = joblib.load('feature_names.pkl')
        return cb, xgb_m, lgbm, ab, feats, []
    except Exception as e:
        # 如果加载出错，返回错误信息（防止程序直接崩溃）
        return None, None, None, None, None, [str(e)]

# ==========================================
# 5. Streamlit 主程序
# ==========================================
def main():
    st.set_page_config(page_title="污水处理厂智能监控与决策支持系统", layout="wide", page_icon="🌊")

    st.title("🌊 污水处理厂智能监控与决策支持系统")
    st.markdown("**EFTM = Ensemble of Four Tree Models** (CatBoost + XGBoost + LightGBM + AdaBoost)")
    st.markdown("---")
    
    # 友情提示：如果在云端看到乱码
    if not os.path.exists('SimHei.ttf'):
        st.info("💡 提示：如果图表中文显示乱码，请将 'SimHei.ttf' 字体文件上传到 GitHub 仓库根目录。")

    # 1. 加载模型
    with st.spinner('正在加载模型文件...'):
        cb_model, xgb_model, lgb_model, ab_model, feature_names, missing_files = load_models()

    if missing_files:
        st.error("❌ 启动失败：找不到以下模型文件")
        st.code('\n'.join(missing_files))
        st.warning("⚠️ 请确保所有 .pkl 文件已上传到 GitHub！")
        st.stop()

    # 初始化 EFTM 权重
    eftm_model = EFTMModel()

    # 2. 侧边栏输入
    st.sidebar.header("🎛️ 实时工况输入")
    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ 进水与时间", "2️⃣ 厌氧池", "3️⃣ 缺氧池", "4️⃣ 好氧池"])
    input_data = {}

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            input_date = st.date_input("预测日期", datetime.now())
            input_time = st.time_input("预测时间", datetime.now())
        with col2:
            input_data['进水量'] = st.number_input("进水量 (m³/h)", value=1000.0)

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1: input_data['厌氧池北溶解氧'] = st.number_input("厌氧池北溶解氧", value=0.2)
        with c2: input_data['厌氧池南ORP'] = st.number_input("厌氧池南ORP", value=-400.0)
        with c3: input_data['厌氧池北ORP'] = st.number_input("厌氧池北ORP", value=-400.0)

    with tab3:
        input_data['缺氧池南污泥浓度'] = st.number_input("缺氧池南污泥浓度", value=3000.0)

    with tab4:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            input_data['好氧池南溶解氧'] = st.number_input("好氧池南溶解氧", value=2.0)
            input_data['好氧池南ORP'] = st.number_input("好氧池南ORP", value=100.0)
        with c2:
            input_data['好氧池北ORP'] = st.number_input("好氧池北ORP", value=100.0)
            input_data['好氧池南污泥浓度'] = st.number_input("好氧池南污泥浓度", value=3000.0)
        with c3:
            input_data['好氧池北污泥浓度'] = st.number_input("好氧池北污泥浓度", value=3000.0)
            input_data['好氧池南PH'] = st.number_input("好氧池南PH", value=7.0)
        with c4:
            input_data['好氧池北PH'] = st.number_input("好氧池北PH", value=7.0)

    # 3. 预测逻辑
    st.markdown("---")
    if st.button("🚀 开始智能预测", type="primary", use_container_width=True):
        # 时间特征处理
        full_dt = datetime.combine(input_date, input_time)
        time_feats = {
            'month_sin': np.sin(2 * np.pi * full_dt.month / 12),
            'month_cos': np.cos(2 * np.pi * full_dt.month / 12),
            'day_sin': np.sin(2 * np.pi * full_dt.day / 31),
            'day_cos': np.cos(2 * np.pi * full_dt.day / 31),
            'hour_sin': np.sin(2 * np.pi * full_dt.hour / 24),
            'hour_cos': np.cos(2 * np.pi * full_dt.hour / 24),
        }

        # 构造输入并对齐
        try:
            input_df = pd.DataFrame([{**input_data, **time_feats}])
            input_df = input_df[feature_names]  # 关键：对齐列顺序
        except KeyError as e:
            st.error(f"❌ 参数缺失: {e}")
            st.stop()

        # 预测
        p_cb = cb_model.predict(input_df)[0]
        p_xgb = xgb_model.predict(input_df)[0]
        p_lgbm = lgb_model.predict(input_df)[0]
        p_ab = ab_model.predict(input_df)[0]
        p_final = eftm_model.predict(p_cb, p_xgb, p_lgbm, p_ab)

        # ----------------------------------
        # E. 结果可视化展示
        # ----------------------------------
        st.success("✅ 预测计算完成！")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("🎯 预测 DO 值", f"{p_final:.4f} mg/L")
            st.info("💡 决策建议：\n根据当前工况，出水指标预期稳定。")

        with c2:
            fig, ax = plt.subplots(figsize=(8, 4))
            models = ['CatBoost', 'XGBoost', 'LightGBM', 'AdaBoost', 'EFTM (最终)']
            vals = [p_cb, p_xgb, p_lgbm, p_ab, p_final]

            # 配色方案
            colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#d62728']

            ax.barh(models, vals, color=colors)
            ax.set_title('各模型预测结果贡献分析', fontsize=14, fontweight='bold')
            ax.set_xlabel('预测值 (mg/L)', fontsize=12)

            # 添加数值标签
            for i, v in enumerate(vals):
                ax.text(v, i, f' {v:.3f}', va='center', fontweight='bold')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)

        # 权重表格展示
        st.markdown("")
        with st.expander("📊 点击查看模型权重详情 (Weight Analysis)", expanded=True):
            weight_df = pd.DataFrame({
                '模型组件 (Model)': ['CatBoost', 'XGBoost', 'LightGBM', 'AdaBoost'],
                '设定权重 (Weight)': [eftm_model.w_cb, eftm_model.w_xgb, eftm_model.w_lgbm, eftm_model.w_ab],
                '独立预测值 (Value)': [p_cb, p_xgb, p_lgbm, p_ab]
            })
            st.table(weight_df.style.format("{:.4f}", subset=['设定权重 (Weight)', '独立预测值 (Value)']))

if __name__ == "__main__":
    main()

