import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from docx import Document
from io import BytesIO
import time

# --- 页面全局配置 ---
st.set_page_config(page_title="路面养护决策系统", layout="wide", initial_sidebar_state="expanded")

# --- 模拟 PSO-XGBoost 模型核心算法逻辑 ---
def mock_pso_xgboost_predict(df):
    """
    模拟逻辑：PCI（破损）、RQI（平整度）、PSSI（强度）越低，预测评分越低。
    实际上这里应该是加载训练好的 XGBoost 模型。
    """
    # 简单的线性加权模拟模型推理
    prediction = (df['PCI'] * 0.4 + df['RQI'] * 0.3 + df['PSSI'] * 0.3) - (df['交通量'] / 2000) - (df['路龄'] * 0.5)
    return np.clip(prediction, 0, 100) # 限制在0-100

# --- 自定义样式 ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .p0-badge { color: red; font-weight: bold; }
    .p1-badge { color: orange; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 初始化 Session State (存储跨页面数据) ---
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# --- 侧边栏导航 ---
st.sidebar.title("🛣️ 路面养护决策系统")
st.sidebar.info("遵循：数据层 → 模型层 → 规则层 → 优化层 → 输出层")
menu = st.sidebar.radio("核心功能模块", 
    ["M1 数据管理模块 (P0)", 
     "M2 路面性能预测模块 (P0)", 
     "M3 养护决策模块 (P0)", 
     "M4 方案生成模块 (P1)", 
     "M5 效果评估模块 (P1)"])

# --- M1 数据管理模块 ---
if menu == "M1 数据管理模块 (P0)":
    st.markdown("## 📂 M1 数据管理模块 <span class='p0-badge'>[P0]</span>", unsafe_allow_html=True)
    st.subheader("多源数据接入、处理与存储")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_file = st.file_uploader("上传路面检测数据 (CSV/Excel)", type=["csv", "xlsx"])
        if st.button("生成示例测试数据"):
            # 自动创建一个符合格式的 DataFrame
            demo_data = pd.DataFrame({
                '路段ID': [f"S{i:03d}" for i in range(101, 111)],
                'PCI': [95, 88, 72, 60, 45, 92, 85, 78, 65, 30],
                'RQI': [92, 85, 70, 58, 42, 90, 82, 75, 60, 25],
                'PSSI': [90, 87, 75, 62, 48, 88, 84, 77, 63, 35],
                '交通量': [2000, 4500, 12000, 15000, 18000, 3000, 5500, 10000, 14000, 22000],
                '路龄': [1, 3, 7, 10, 14, 2, 4, 6, 9, 15]
            })
            st.session_state.raw_data = demo_data
            st.success("示例数据已生成！")

    if uploaded_file:
        st.session_state.raw_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success("文件上传完成")

    with col2:
        if st.session_state.raw_data is not None:
            st.write("### 实时数据预览")
            st.dataframe(st.session_state.raw_data, use_container_width=True)

# --- M2 路面性能预测模块 ---
elif menu == "M2 路面性能预测模块 (P0)":
    st.markdown("## 📈 M2 路面性能预测模块 <span class='p0-badge'>[P0]</span>", unsafe_allow_html=True)
    st.subheader("PSO-XGBoost 模型推理与可视化")
    
    if st.session_state.raw_data is None:
        st.warning("⚠️ 请先在 M1 模块上传或生成数据")
    else:
        if st.button("🚀 启动 PSO 优化并运行 XGBoost 推理"):
            with st.spinner('正在进行粒子群寻优参数...'):
                time.sleep(1.5) # 模拟计算耗时
                df = st.session_state.raw_data.copy()
                df['预测健康分'] = mock_pso_xgboost_predict(df)
                st.session_state.processed_data = df
                st.success("模型推理完成！")

        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(df, x="路段ID", y="预测健康分", color="预测健康分", 
                             title="各路段预测健康指数", color_continuous_scale='RdYlGn')
                st.plotly_chart(fig)
            with c2:
                fig2 = px.scatter(df, x="交通量", y="预测健康分", size="路龄", hover_name="路段ID",
                                 title="交通量、路龄与评分关系图")
                st.plotly_chart(fig2)

# --- M3 养护决策模块 ---
elif menu == "M3 养护决策模块 (P0)":
    st.markdown("## 🧠 M3 养护决策模块 <span class='p0-badge'>[P0]</span>", unsafe_allow_html=True)
    st.subheader("智能决策生成与方案优化")

    if st.session_state.processed_data is None:
        st.warning("⚠️ 请先在 M2 模块完成性能预测")
    else:
        df = st.session_state.processed_data
        
        # 决策逻辑函数
        def make_decision(score):
            if score >= 85: return "日常养护", "进行常规清扫和局部小补"
            elif 70 <= score < 85: return "预防性养护", "建议进行精表处或碎石封层"
            elif 55 <= score < 70: return "修复性养护", "建议进行 4cm 沥青罩面"
            else: return "结构大修", "建议进行全深层铣刨重新铺筑"

        df[['决策方案', '技术细节']] = df.apply(lambda x: make_decision(x['预测健康分']), axis=1, result_type='expand')
        st.session_state.processed_data = df
        
        st.write("### 自动生成的决策矩阵")
        st.dataframe(df[['路段ID', '预测健康分', '决策方案', '技术细节']], use_container_width=True)

# --- M4 方案生成模块 ---
elif menu == "M4 方案生成模块 (P1)":
    st.markdown("## 📄 M4 方案生成模块 <span class='p1-badge'>[P1]</span>", unsafe_allow_html=True)
    st.subheader("标准化养护方案文档自动生成")

    if st.session_state.processed_data is None or '决策方案' not in st.session_state.processed_data.columns:
        st.warning("⚠️ 请先完成前三个模块的操作")
    else:
        df = st.session_state.processed_data
        selected_id = st.selectbox("选择要导出方案的路段", df['路段ID'])
        row = df[df['路段ID'] == selected_id].iloc[0]

        st.info(f"选定路段：{selected_id} | 建议方案：{row['决策方案']}")

        if st.button("生成 Word 技术方案"):
            doc = Document()
            doc.add_heading('公路养护技术方案', 0)
            doc.add_paragraph(f'路段编号: {selected_id}')
            doc.add_paragraph(f'当前预测评分: {row["预测健康分"]:.2f}')
            doc.add_paragraph(f'养护等级: {row["决策方案"]}')
            doc.add_heading('技术实施要求', level=1)
            doc.add_paragraph(row['技术细节'])
            doc.add_paragraph("1. 施工前需进行现场交通管制。")
            doc.add_paragraph("2. 严格遵循《公路养护技术规范》要求。")
            
            buffer = BytesIO()
            doc.save(buffer)
            st.download_button(
                label="📥 点击下载 Word 文档",
                data=buffer.getvalue(),
                file_name=f"方案_{selected_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

# --- M5 效果评估模块 ---
elif menu == "M5 效果评估模块 (P1)":
    st.markdown("## 📊 M5 效果评估模块 <span class='p1-badge'>[P1]</span>", unsafe_allow_html=True)
    st.subheader("养护后跟踪评估与反馈")

    st.write("### 养护前后性能对比模拟")
    # 模拟对比数据
    compare_data = pd.DataFrame({
        '阶段': ['养护前', '养护后'],
        'PCI (破损指数)': [65, 98],
        'RQI (平整度)': [60, 95],
        '安全系数': [0.7, 0.95]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[65, 60, 0.7*100], theta=['PCI','RQI','安全'], fill='toself', name='养护前'))
    fig.add_trace(go.Scatterpolar(r=[98, 95, 0.95*100], theta=['PCI','RQI','安全'], fill='toself', name='养护后'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="路段改善雷达图")
    st.plotly_chart(fig)
    st.success("评估结果：养护后综合性能提升了 45%！")