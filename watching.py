import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request  # 用于自动下载字体

# ==========================================
# 1. 基础配置与路径修复
# ==========================================
try:
    # 强制将工作目录切换到当前脚本所在的文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except:
    pass

# ==========================================
# 2. 核心修复：自动下载并加载中文字体
# ==========================================
def set_chinese_font():
    font_filename = 'SimHei.ttf'
    
    # 如果当前文件夹里没有字体文件，就自动去网上下载一个
    if not os.path.exists(font_filename):
        with st.spinner("正在为云端环境下载中文字体，请稍候..."):
            try:
                # 这是一个公开的 SimHei 字体下载链接
                url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
                urllib.request.urlretrieve(url, font_filename)
                st.success("✅ 字体下载成功！")
            except Exception as e:
                st.error(f"字体下载失败，图表可能无法显示中文。错误: {e}")
                return

    # 加载字体
    if os.path.exists(font_filename):
        fm.fontManager.addfont(font_filename)
        plt.rcParams['font.sans-serif'] = ['SimHei']
    else:
        # 如果实在没有，回退到系统默认
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    
    # 解决负号显示问题
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
    st.set_page_config(page_title="污水厂水质预测系统", layout="wide", page_icon="🌊")

    st.title("🌊 污水处理厂出水水质预测系统")
    st.markdown("**EFTM = Ensemble of Four Tree Models** (CatBoost + XGBoost + LightGBM + AdaBoost)")
    st.markdown("---")
    
    # 1. 加载模型
    with st.spinner('正在加载模型文件...'):
        cb_model, xgb_model, lgb_model, ab_model, feature_names, missing_files = load_models()

    if missing_files:
        st.error("❌ 启动失败：找不到以下模型文件")
        st.code('\n'.join(missing_files))
        st.warning("⚠️ 请确保所有 .pkl 文件已上传到 GitHub！")
        st.stop()
    
    # 如果模型加载失败（比如版本不兼容严重报错）
    if cb_model is None:
        st.error(f"模型加载出错: {missing_files[0] if missing_files else '未知错误'}")
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
        try:
            p_cb = cb_model.predict(input_df)[0]
            p_xgb = xgb_model.predict(input_df)[0]
            p_lgbm = lgb_model.predict(input_df)[0]
            p_ab = ab_model.predict(input_df)[0]
            p_final = eftm_model.predict(p_cb, p_xgb, p_lgbm, p_ab)
        except Exception as e:
            st.error(f"预测计算出错: {e}")
            st.stop()

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
            models = ['CatBoost', 'XGBoost', 'LightGBM', 'AdaBoost', 'EFTM (Final)']
            vals = [p_cb, p_xgb, p_lgbm, p_ab, p_final]

            # 配色方案
            colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#d62728']

            ax.barh(models, vals, color=colors)
            ax.set_title('各模型预测结果贡献分析', fontsize=14, fontweight='bold')
            ax.set_xlabel('预测值 DO (mg/L)', fontsize=12)

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
