import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from docx import Document
from io import BytesIO
import time
import os
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
# 规则层所需：Apriori 与 PLSR
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# --- 页面全局配置 ---
st.set_page_config(page_title="路面养护决策系统", layout="wide", initial_sidebar_state="expanded")

# --- 专家知识库: 广西对策库示例 ---
EXPERT_KNOWLEDGE = {
    "雾封层": {
        "场景": "轻度龟裂、表层松散、抗滑性能轻度衰减。PCI≥85，无结构性病害。",
        "参数": "沥青用量0.3~0.5kg/m²；施工环境温度≥10℃。采用广西本地生产的高黏改性乳化沥青。",
        "效果": "PCI提升5~8，预期使用寿命5~6年。",
        "窗口期": "10月~次年4月"
    },
    "同步碎石封层": {
        "场景": "轻度车辙(RD≤5mm)、路面渗水、轻度龟裂。PCI≥80, PSSI≥80。",
        "参数": "沥青用量1.2~1.5kg/m²，碎石撒布覆盖率100%~120%。采用SBS改性沥青(软化点≥60℃)。",
        "效果": "PCI提升8~12，RD降低2~3mm，使用寿命6~8年。",
        "窗口期": "10月~次年3月"
    },
    "局部铣刨重铺": {
        "场景": "中度龟裂、松散、坑槽。70≤PCI<80, PSSI≥75。",
        "参数": "铣刨深度2~4cm，压实度≥96%。采用AC-13C改性沥青混合料。马歇尔稳定度≥10kN。",
        "效果": "PCI提升15~20，病害完全修复，使用寿命8~10年。",
        "窗口期": "10月~次年4月"
    },
    "结构大修": {
        "场景": "PCI<70 或结构强度严重不足。",
        "参数": "全深层铣刨重新铺筑，基层加固。",
        "效果": "彻底解决结构性病害，恢复路面全寿命周期。",
        "窗口期": "全年(避开雨季)"
    }
}

# --- 模拟 PSO-XGBoost 模型核心算法逻辑 ---
def forecast_pci_pssi(df, years=[0,1,2,3,5]):
    """
    按年份预测 PCI 和 PSSI 的简单模拟函数。
    返回一个扩展后的 DataFrame，包含每个路段每年预测的 `预测PCI` 和 `预测PSSI`。
    预测逻辑：基于初始 PCI/PSSI，加上一个由交通量和当前路龄决定的年降幅。
    """
    rows = []
    for _, r in df.iterrows():
        base_pci = float(r.get('PCI', 0))
        base_pssi = float(r.get('PSSI', 0))
        traffic = float(r.get('交通量', 0))
        age = float(r.get('路龄', 0))

        # 年降幅基准（可根据实际模型调整）
        # 交通量越大、路龄越长，降幅越高
        annual_degrade_pci = 1.5 + (traffic / 30000.0) * 6.0 + (age / 20.0) * 2.0
        annual_degrade_pssi = 1.0 + (traffic / 40000.0) * 4.0 + (age / 25.0) * 1.5

        for y in years:
            pred_pci = max(0.0, base_pci - y * annual_degrade_pci)
            pred_pssi = max(0.0, base_pssi - y * annual_degrade_pssi)
            rows.append({
                '路段ID': r.get('路段ID'),
                '年': int(y),
                '预测PCI': round(pred_pci, 2),
                '预测PSSI': round(pred_pssi, 2),
                '交通量': traffic,
                '路龄': age
            })

    return pd.DataFrame(rows)


def compute_pci_pssi_from_measurements(df1, df2):
    """
    将两个按序号对齐的检测表合并并计算 PCI 与 PSSI 的估计值。
    df1: 平整度和构造深度（含 左/右平整度、构造深度、跳车 等）
    df2: 车辙（含 左/右车辙）

    返回：DataFrame，包含列 ['路段ID', 'PCI', 'PSSI', '交通量', '路龄']（若无交通量/路龄则填默认）
    计算规则（简单启发式）：
      - 平整度、跳车、车辙 越大，PCI 越低；归一化为 0-100。
      - 车辙与构造深度影响 PSSI（强度），车辙越深/构造深度越大表示承载能力下降。
    这些是示例启发式公式，建议替换为真实模型。
    """
    # 选择用于合并的键：优先使用 序号 或 起止桩号组合；如果有 '序号' 则用之
    left = df1.copy()
    right = df2.copy()

    if '序号' in left.columns and '序号' in right.columns:
        merged = left.merge(right, on='序号', suffixes=('_p', '_r'))
    elif '起点桩号' in left.columns and '起点桩号' in right.columns:
        merged = left.merge(right, on=['起点桩号','终止桩号'], suffixes=('_p', '_r'))
    else:
        # 兜底：按行索引对齐
        merged = pd.concat([left.reset_index(drop=True), right.reset_index(drop=True)], axis=1)

    # 构造一些指标
    def safe_mean(vals):
        vals = [float(v) for v in vals if pd.notna(v)]
        return sum(vals)/len(vals) if vals else 0.0

    rows = []
    for _, r in merged.iterrows():
        # 平整度参考左/右平整度列名可能为 左平整度/右平整度 或 left/right 平整度
        lp = r.get('左平整度', r.get('left_flatness', np.nan))
        rp = r.get('右平整度', r.get('right_flatness', np.nan))
        left_jump = r.get('左路面跳车', r.get('left_jump', 0))
        right_jump = r.get('右路面跳车', r.get('right_jump', 0))
        rut_l = r.get('左车辙', r.get('左车辙', r.get('left_rut', np.nan)))
        rut_r = r.get('右车辙', r.get('右车辙', r.get('right_rut', np.nan)))
        struct_left = r.get('左构造深度', r.get('left_profile', np.nan))
        struct_right = r.get('右构造深度', r.get('right_profile', np.nan))

        mean_flat = safe_mean([lp, rp])
        mean_jump = safe_mean([left_jump, right_jump])
        mean_rut = safe_mean([rut_l, rut_r])
        mean_struct = safe_mean([struct_left, struct_right])

        # PCI 基本上与平整度正相关，与跳车和车辙负相关
        # 归一化：假设平整度理想值 ~0-5（示例），跳车/车辙越小越好 -> 将它们映射到扣分
        pci_score = 100.0
        # 平整度较小（好），这里把平整度乘以负面系数
        pci_score -= mean_flat * 2.0
        pci_score -= mean_jump * 0.8
        pci_score -= mean_rut * 1.2
        # 结构深度过大也扣分
        pci_score -= max(0.0, mean_struct - 0.5) * 10.0

        # PSSI 主要受车辙和构造深度影响，越大表明结构问题，PSSI 越低
        pssi_score = 100.0 - mean_rut * 2.5 - max(0.0, mean_struct - 0.5) * 8.0 - mean_jump * 0.5

        pci_score = float(np.clip(pci_score, 0, 100))
        pssi_score = float(np.clip(pssi_score, 0, 100))

        rows.append({
            '路段ID': r.get('序号', r.get('起点桩号', None)) or f"ID_{_}",
            'PCI': round(pci_score,2),
            'PSSI': round(pssi_score,2),
            # 保留交通量、路龄列（若存在），否则设为默认
            '交通量': r.get('交通量', r.get('traffic', 5000)),
            '路龄': r.get('路龄', r.get('age', 5))
        })

    return pd.DataFrame(rows)


def _build_features_from_measurements(df_meas_rut):
    # 输入为已合并的测量表（含左/右平整度、跳车、车辙、构造深度、交通量、路龄）
    rows = []
    for _, r in df_meas_rut.iterrows():
        lp = r.get('左平整度', np.nan)
        rp = r.get('右平整度', np.nan)
        left_jump = r.get('左路面跳车', 0)
        right_jump = r.get('右路面跳车', 0)
        rut_l = r.get('左车辙', np.nan)
        rut_r = r.get('右车辙', np.nan)
        struct_left = r.get('左构造深度', np.nan)
        struct_right = r.get('右构造深度', np.nan)

        mean_flat = np.nanmean([v for v in [lp, rp] if pd.notna(v)]) if any(pd.notna(v) for v in [lp, rp]) else 0.0
        mean_jump = np.nanmean([v for v in [left_jump, right_jump] if pd.notna(v)]) if any(pd.notna(v) for v in [left_jump, right_jump]) else 0.0
        mean_rut = np.nanmean([v for v in [rut_l, rut_r] if pd.notna(v)]) if any(pd.notna(v) for v in [rut_l, rut_r]) else 0.0
        mean_struct = np.nanmean([v for v in [struct_left, struct_right] if pd.notna(v)]) if any(pd.notna(v) for v in [struct_left, struct_right]) else 0.0

        rows.append({
            'mean_flat': mean_flat,
            'mean_jump': mean_jump,
            'mean_rut': mean_rut,
            'mean_struct': mean_struct,
            '交通量': r.get('交通量', 5000),
            '路龄': r.get('路龄', 5)
        })
    return pd.DataFrame(rows)


def detect_and_remove_bad_rows(df, method='iqr', z_thresh=3.0, iqr_k=1.5, nan_thresh=0.5, domain_rules=True):
    """
    清洗数据：
    - domain_rules: 基于常识的字段范围检测（如 PCI/PSSI 0-100，交通量非负等）。
    - method: 'iqr' 或 'zscore'，用于数值列异常值检测并剔除。
    - nan_thresh: 行中缺失值比例超过阈值则剔除。
    返回 (cleaned_df, removed_df, report)
    """
    if df is None or df.empty:
        return df, pd.DataFrame(), {'reason': 'empty'}

    data = df.copy()
    removed_mask = pd.Series(False, index=data.index)

    # 1) 删除缺失比例过高的行
    row_nan_frac = data.isna().mean(axis=1)
    nan_remove = row_nan_frac > float(nan_thresh)
    removed_mask = removed_mask | nan_remove

    # 2) 域规则检测
    if domain_rules:
        # PCI/PSSI 在 0-100
        for col in ['PCI','PSSI','预测PCI','预测PSSI']:
            if col in data.columns:
                bad = (data[col] < 0) | (data[col] > 100) | data[col].isna()
                removed_mask = removed_mask | bad
        # 交通量/路龄/车辙/构造深度 合理范围
        if '交通量' in data.columns:
            bad = (data['交通量'] < 0) | (data['交通量'] > 1e7) | data['交通量'].isna()
            removed_mask = removed_mask | bad
        if '路龄' in data.columns:
            bad = (data['路龄'] < 0) | (data['路龄'] > 200) | data['路龄'].isna()
            removed_mask = removed_mask | bad

    # 3) 数值异常检测（IQR 或 Z-score）
    num_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    if num_cols:
        sub = data[num_cols].copy()
        # 替换 inf 为 NaN
        sub.replace([np.inf, -np.inf], np.nan, inplace=True)
        if method == 'zscore':
            # 计算 zscore
            mean = sub.mean()
            std = sub.std().replace(0, np.nan)
            z = (sub - mean) / std
            z_bad = z.abs() > float(z_thresh)
            z_any = z_bad.any(axis=1)
            removed_mask = removed_mask | z_any.fillna(False)
        else:
            # IQR 方法
            Q1 = sub.quantile(0.25)
            Q3 = sub.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - float(iqr_k) * IQR
            upper = Q3 + float(iqr_k) * IQR
            out_lower = (sub < lower)
            out_upper = (sub > upper)
            out_any = (out_lower | out_upper).any(axis=1)
            removed_mask = removed_mask | out_any.fillna(False)

    cleaned = data.loc[~removed_mask].copy()
    removed = data.loc[removed_mask].copy()
    report = {
        'original_rows': len(data),
        'removed_rows': int(removed_mask.sum()),
        'cleaned_rows': int((~removed_mask).sum())
    }
    return cleaned, removed, report


def simple_pso_optimize_xgb(X, y, n_particles=10, n_iters=10, random_state=42, augment=False, aug_std_fraction=0.01):
    """
    改进后的 PSO 用于 XGBoost 超参调优。
    - 增加粒子数量和迭代次数。
    - 调整参数范围。
    - 启用早停机制。
    - 增加诊断信息。
    """
    np.random.seed(random_state)

    # 参数边界（扩大搜索空间）
    bounds = {
        'max_depth': (3, 12),
        'learning_rate': (0.001, 0.2),
        'n_estimators': (50, 500),
        'subsample': (0.5, 1.0)
    }

    # 初始化粒子位置与速度
    particles = []
    velocities = []
    scores = []
    for _ in range(n_particles):
        p = {
            'max_depth': np.random.uniform(*bounds['max_depth']),
            'learning_rate': np.random.uniform(*bounds['learning_rate']),
            'n_estimators': np.random.uniform(*bounds['n_estimators']),
            'subsample': np.random.uniform(*bounds['subsample'])
        }
        v = {k: 0.0 for k in p}
        particles.append(p)
        velocities.append(v)
        scores.append(1e9)

    # 记录个体最优与全局最优
    pbest = [p.copy() for p in particles]
    pbest_scores = scores.copy()
    gbest = None
    gbest_score = 1e9

    def eval_params(param_dict):
        # 转换为 int/float
        params = {
            'max_depth': int(round(param_dict['max_depth'])),
            'learning_rate': float(param_dict['learning_rate']),
            'n_estimators': int(round(param_dict['n_estimators'])),
            'subsample': float(param_dict['subsample'])
        }
        # 5-fold CV RMSE
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        rmses = []
        for tr_idx, te_idx in kf.split(X):
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
            # 若启用数据增强，只对训练集加入微小高斯噪声（按列 std 比例）以增加样本差异
            if augment:
                try:
                    stds = Xtr.std().replace(0, 1.0)
                    scale = (aug_std_fraction * stds).values
                    noise = np.random.randn(*Xtr.shape) * scale
                    Xtr_noisy = pd.DataFrame(Xtr.values + noise, columns=Xtr.columns, index=Xtr.index)
                    Xtr = Xtr_noisy
                except Exception:
                    pass
            # 加速：使用 hist 树方法并启用多线程
            model = xgb.XGBRegressor(
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                subsample=params['subsample'],
                objective='reg:squarederror',
                verbosity=0,
                n_jobs=-1,
                tree_method='hist',
                early_stopping_rounds=10,
                eval_metric='rmse'
            )
            # 兼容不同 xgboost 版本：直接训练
            model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
            pred = model.predict(Xte)
            rmses.append(math.sqrt(mean_squared_error(yte, pred)))
        return float(np.mean(rmses))

    w = 0.5
    c1 = 0.8
    c2 = 0.9

    for it in range(n_iters):
        for i, p in enumerate(particles):
            score = eval_params(p)
            # 更新个体最优
            if score < pbest_scores[i]:
                pbest[i] = p.copy()
                pbest_scores[i] = score
            # 更新全局最优
            if score < gbest_score:
                gbest = p.copy()
                gbest_score = score

        # 输出诊断信息
        print(f"Iteration {it+1}/{n_iters}, Best Score: {gbest_score:.4f}")

        # 更新速度与位置
        for i in range(n_particles):
            for k in particles[i].keys():
                r1 = np.random.rand()
                r2 = np.random.rand()
                velocities[i][k] = w * velocities[i][k] + c1 * r1 * (pbest[i][k] - particles[i][k]) + c2 * r2 * (gbest[k] - particles[i][k])
                particles[i][k] = particles[i][k] + velocities[i][k]
                # 边界约束
                low, high = bounds[k]
                particles[i][k] = float(np.clip(particles[i][k], low, high))

    best_params = {
        'max_depth': int(round(gbest['max_depth'])),
        'learning_rate': float(gbest['learning_rate']),
        'n_estimators': int(round(gbest['n_estimators'])),
        'subsample': float(gbest['subsample'])
    }
    return best_params, gbest_score

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
     "R1 规则层 (P1)",
     "M3 养护决策模块 (P0)", 
     "M4 方案生成模块 (P1)"])

# --- M1 数据管理模块 ---
if menu == "M1 数据管理模块 (P0)":
    st.markdown("## 📂 M1 数据管理模块 <span class='p0-badge'>[P0]</span>", unsafe_allow_html=True)
    st.subheader("多源数据接入、处理与存储")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**上传：平整度和构造深度 (CSV / XLSX)**")
        file_meas = st.file_uploader("上传 平整度和构造深度 (CSV 或 XLSX)", type=["csv","xls","xlsx"], key='meas')
        st.markdown("**上传：车辙 (CSV)**")
        file_rut = st.file_uploader("上传 车辙 (CSV 或 XLSX)", type=["csv","xls","xlsx"], key='rut')

        if st.button("生成示例测试数据（测量表 + 车辙表）"):
            # 生成示例的测量表与车辙表，并计算出 PCI/PSSI
            demo_meas = pd.DataFrame({
                '序号': [i for i in range(1,11)],
                '起点桩号': [179080 + 10*(i-1) for i in range(1,11)],
                '终止桩号': [179080 + 10*i for i in range(1,11)],
                '左平整度': [20.49,2.11,1.73,2.48,3.91,2.59,1.68,1.49,2.08,2.06],
                '右平整度': [19.77,2.68,3.85,2.46,4.84,2.52,1.96,1.85,2.04,2.48],
                '左构造深度': [0.7,0.44,0.45,0.48,0.52,0.41,0.39,0.46,0.45,0.48],
                '右构造深度': [1.09,0.67,0.61,1.03,0.75,0.47,0.43,0.52,0.52,0.45],
                '中构造深度': [1.01,0.48,0.44,0.7,0.63,0.36,0.37,0.41,0.34,0.35],
                '左路面跳车': [146.84,13.18,7.24,9.94,9.51,13.17,4.83,7.38,5.95,6.12],
                '右路面跳车': [61.91,8.87,13.28,4.43,18.38,7.34,3.95,9.71,7.6,8.18],
                '车速': [10.1,13,15.8,19.1,23,24.8,28.1,29.5,30.6,31.7]
            })
            demo_rut = pd.DataFrame({
                '序号': [i for i in range(1,11)],
                '起点桩号': demo_meas['起点桩号'],
                '终止桩号': demo_meas['终止桩号'],
                '左车辙': [15.19,2.21,4.92,2.62,1.56,3.35,1.66,1.43,1.53,1.61],
                '右车辙': [18.61,3.55,5.45,4.97,4.09,3.88,3.3,3.27,4.23,3.65]
            })
            merged_demo = compute_pci_pssi_from_measurements(demo_meas, demo_rut)
            # 保存到 session 以便预览
            st.session_state.meas_raw = demo_meas
            st.session_state.rut_raw = demo_rut
            st.session_state.raw_data = merged_demo
            st.success("示例测量表与车辙表已生成并计算 PCI/PSSI！")

        if file_meas:
            try:
                fname = getattr(file_meas, 'name', '')
                if fname.lower().endswith(('.xls', '.xlsx')):
                    df_meas = pd.read_excel(file_meas)
                else:
                    df_meas = pd.read_csv(file_meas)
                st.session_state.meas_raw = df_meas
                st.success("平整度/构造深度表已上传并解析")
            except Exception as e:
                st.error(f"平整度表解析失败: {e}")

        if file_rut:
            try:
                fname = getattr(file_rut, 'name', '')
                if fname.lower().endswith(('.xls', '.xlsx')):
                    df_rut = pd.read_excel(file_rut)
                else:
                    df_rut = pd.read_csv(file_rut)
                st.session_state.rut_raw = df_rut
                st.success("车辙表已上传并解析")
            except Exception as e:
                st.error(f"车辙表解析失败: {e}")

        # 若同时存在 meas_raw 与 rut_raw，自动合并计算
        if 'meas_raw' in st.session_state and 'rut_raw' in st.session_state:
            try:
                merged = compute_pci_pssi_from_measurements(st.session_state.meas_raw, st.session_state.rut_raw)
                st.session_state.raw_data = merged
                st.success("已基于上传两表计算 PCI/PSSI 并生成预览")
            except Exception as e:
                st.error(f"合并计算失败: {e}")
        elif 'meas_raw' in st.session_state and 'rut_raw' not in st.session_state:
            # 仅测量表，尝试计算但告警
            try:
                merged = compute_pci_pssi_from_measurements(st.session_state.meas_raw, pd.DataFrame())
                st.session_state.raw_data = merged
                st.warning('仅上传测量表，已基于测量表计算 PCI/PSSI（可能不完整）')
            except Exception as e:
                st.error(f"基于测量表计算失败: {e}")

    with col2:
        st.write("### 实时数据预览")
        if 'meas_raw' in st.session_state:
            st.write("**平整度与构造深度 表（预览）**")
            st.dataframe(st.session_state.meas_raw, use_container_width=True)
        if 'rut_raw' in st.session_state:
            st.write("**车辙 表（预览）**")
            st.dataframe(st.session_state.rut_raw, use_container_width=True)
        if st.session_state.raw_data is not None:
            st.write("**计算得到的 PCI / PSSI 结果（预览）**")
            st.dataframe(st.session_state.raw_data, use_container_width=True)

        # --- 数据清洗交互 ---
        st.markdown("### 数据清洗（剔除坏数）")
        st.write("可以使用域规则 + 统计异常检测（IQR / Z-score）剔除坏数据，建议先预览再应用。")
        do_clean = st.checkbox("启用数据清洗工具（在上传后运行）", key='enable_clean_tool')
        if do_clean:
            col_a, col_b = st.columns([1,1])
            with col_a:
                method = st.selectbox("异常检测方法", options=['iqr','zscore'], index=0, key='clean_method')
                iqr_k = st.number_input("IQR multiplier (k)", value=1.5, step=0.1, key='clean_iqr_k')
                z_thresh = st.number_input("Z-score 阈值", value=3.0, step=0.1, key='clean_z_thr')
            with col_b:
                nan_thresh = st.slider("行缺失比例阈值 (大于该阈值将被删除)", 0.0, 1.0, 0.5, key='clean_nan_thr')
                domain_rules = st.checkbox("启用域规则检测 (PCI 0-100, 交通量非负等)", value=True, key='clean_domain')

            if st.button("运行清洗并预览（仅预览，不修改原始数据）", key='run_preview_clean'):
                df_tmp = st.session_state.raw_data.copy()
                cleaned, removed, report = detect_and_remove_bad_rows(df_tmp, method=method, z_thresh=z_thresh, iqr_k=iqr_k, nan_thresh=nan_thresh, domain_rules=domain_rules)
                st.session_state.raw_data_clean_preview = cleaned
                st.session_state.raw_data_removed_preview = removed
                st.session_state.raw_data_clean_report = report
                st.write(f"清洗报告：原始行数={report['original_rows']}，剔除行数={report['removed_rows']}，保留行数={report['cleaned_rows']}")
                if not removed.empty:
                    st.write('被剔除样例（前20行）：')
                    st.dataframe(removed.head(20))
                else:
                    st.info('未检测到需要剔除的异常行')

            if 'raw_data_clean_preview' in st.session_state:
                if st.button('应用清洗结果（替换原始数据）', key='apply_clean'):
                    # 备份原始数据以便回滚
                    if 'raw_data_original' not in st.session_state:
                        st.session_state.raw_data_original = st.session_state.raw_data.copy()
                    st.session_state.raw_data = st.session_state.raw_data_clean_preview.copy()
                    # 清理预览缓存
                    st.session_state.pop('raw_data_clean_preview', None)
                    st.session_state.pop('raw_data_removed_preview', None)
                    st.success('已将清洗结果应用为原始数据')

            if 'raw_data_original' in st.session_state:
                if st.button('回滚到原始上传数据（撤销清洗）', key='rollback_raw'):
                    st.session_state.raw_data = st.session_state.raw_data_original.copy()
                    st.session_state.pop('raw_data_original', None)
                    st.success('已回滚为原始数据')

# --- M2 路面性能预测模块 ---
elif menu == "M2 路面性能预测模块 (P0)":
    st.markdown("## 📈 M2 路面性能预测模块 <span class='p0-badge'>[P0]</span>", unsafe_allow_html=True)
    st.subheader("按年份预测 PCI / PSSI 并可视化")

    if st.session_state.raw_data is None:
        st.warning("⚠️ 请先在 M1 模块上传或生成数据")
    else:
        # 年份选择（默认 0,1,2,3,5 年）
        years_input = st.text_input("请输入要预测的年份（逗号分隔，例如：0,1,2,3,4,5,6,7,8,9,10）", value="0,1,2,3,4,5,6,7,8,9,10")
        try:
            years = [int(x.strip()) for x in years_input.split(',') if x.strip()!='']
        except:
            years = [0,1,2,3,5]

        if st.button("🚀 运行按年预测 (PCI / PSSI)"):
            with st.spinner('正在模拟按年预测...'):
                time.sleep(1.0)
                df = st.session_state.raw_data.copy()
                # 如果 raw_data 是测量表（未计算 PCI/PSSI），尝试用启发式函数先得到基线
                if 'PCI' not in df.columns or 'PSSI' not in df.columns:
                    # 假设上传的是测量表，已在 M1 用 compute_pci_pssi_from_measurements 生成
                    base_df = df.copy()
                else:
                    base_df = df.copy()
                # 直接基于已有 PCI/PSSI 做年度展开（启发式基线），避免用户误以为需要先运行 PSO
                try:
                    # 若 raw_data 中未包含 PCI/PSSI，则尝试用启发式计算
                    if 'PCI' not in base_df.columns or 'PSSI' not in base_df.columns:
                        base_df_calc = compute_pci_pssi_from_measurements(base_df, pd.DataFrame())
                        base_forcast_src = base_df_calc
                    else:
                        base_forcast_src = base_df
                    forecast_df = forecast_pci_pssi(base_forcast_src, years=years)
                    st.session_state.processed_data = forecast_df
                    st.session_state.pso_models = {}
                    st.session_state.pso_results = {}
                    st.success("按年预测已生成（启发式基线或已有 PCI/PSSI 展开）")
                except Exception as e:
                    st.session_state.processed_data = None
                    st.session_state.pso_models = {}
                    st.session_state.pso_results = {}
                    st.error(f"基线年度预测失败: {e}")

        # PSO-XGBoost 训练区：在已有 raw_data 的基础上训练模型
        if st.button("⚙️ 使用 PSO 优化并训练 XGBoost (适用于大数据)"):
            if st.session_state.raw_data is None:
                st.error('请先在 M1 上传或生成数据')
            else:
                with st.spinner('正在准备训练数据...'):
                    df0 = st.session_state.raw_data.copy()
                    # 若上传的是测量表，df0 可能包含原始测量列；我们需要合并测量以构造特征
                    # 如果数据已经是合并后的特征表（含 mean_*），直接用之
                    feature_df = None
                    if set(['mean_flat','mean_jump','mean_rut','mean_struct']).issubset(df0.columns):
                        feature_df = df0[['mean_flat','mean_jump','mean_rut','mean_struct','交通量','路龄']].copy()
                    else:
                        # 尝试按现有列构建特征
                        feature_df = _build_features_from_measurements(df0)

                    # 目标 y：如果有 PCI/PSSI 列则用原始值，否则用启发式计算的值
                    if 'PCI' in df0.columns and 'PSSI' in df0.columns:
                        y_pci = df0['PCI']
                        y_pssi = df0['PSSI']
                    else:
                        # 使用启发式函数输出
                        tmp = compute_pci_pssi_from_measurements(df0, pd.DataFrame())
                        y_pci = tmp['PCI']
                        y_pssi = tmp['PSSI']

                # 训练 PCI 模型
                # 读取增强/ensemble 配置
                aug_enabled = st.sidebar.checkbox('启用训练数据增强（Jitter）', value=True)
                aug_std_frac = st.sidebar.slider('增强强度：特征 std 的比例', 0.0, 0.2, 0.01)
                ensemble_count = st.sidebar.slider('训练时模型数（Ensemble 平均）', 1, 5, 3)
                with st.spinner('正在用 PSO 优化 XGBoost (PCI)...'):
                    try:
                        best_pci_params, pci_score = simple_pso_optimize_xgb(feature_df, y_pci, n_particles=8, n_iters=6, augment=aug_enabled, aug_std_fraction=aug_std_frac)
                        st.session_state.pso_results['pci'] = (best_pci_params, pci_score)
                        st.success(f"PCI 模型最佳参数: {best_pci_params}  CV RMSE={pci_score:.3f}")
                    except Exception as e:
                        st.error(f"PCI 模型训练失败: {e}")

                # 训练 PSSI 模型
                with st.spinner('正在用 PSO 优化 XGBoost (PSSI)...'):
                    try:
                        best_pssi_params, pssi_score = simple_pso_optimize_xgb(feature_df, y_pssi, n_particles=8, n_iters=6, augment=aug_enabled, aug_std_fraction=aug_std_frac)
                        st.session_state.pso_results['pssi'] = (best_pssi_params, pssi_score)
                        st.success(f"PSSI 模型最佳参数: {best_pssi_params}  CV RMSE={pssi_score:.3f}")
                    except Exception as e:
                        st.error(f"PSSI 模型训练失败: {e}")

                # 用最佳参数训练最终模型并保存
                with st.spinner('训练最终模型并生成年度预测...'):
                    # 训练最终模型：支持增强时的 ensemble 平均
                    def train_ensemble(params, X, y, ensemble_count, aug_enabled, aug_std_frac):
                        if aug_enabled and ensemble_count > 1:
                            models = []
                            for _m in range(ensemble_count):
                                X_aug = X.copy()
                                try:
                                    stds = X_aug.std().replace(0, 1.0)
                                    scale = (aug_std_frac * stds).values
                                    noise = np.random.randn(*X_aug.shape) * scale
                                    X_aug = pd.DataFrame(X_aug.values + noise, columns=X_aug.columns, index=X_aug.index)
                                except Exception:
                                    pass
                                m = xgb.XGBRegressor(**params, objective='reg:squarederror', verbosity=0, n_jobs=-1, tree_method='hist')
                                m.fit(X_aug, y)
                                models.append(m)
                            return models
                        else:
                            m = xgb.XGBRegressor(**params, objective='reg:squarederror', verbosity=0, n_jobs=-1, tree_method='hist')
                            m.fit(X, y)
                            return m

                    model_pci_obj = train_ensemble(best_pci_params, feature_df, y_pci, ensemble_count, aug_enabled, aug_std_frac)
                    model_pssi_obj = train_ensemble(best_pssi_params, feature_df, y_pssi, ensemble_count, aug_enabled, aug_std_frac)
                    st.session_state.pso_models['pci'] = model_pci_obj
                    st.session_state.pso_models['pssi'] = model_pssi_obj

                    # 生成基线预测并按年展开
                    base_pred = feature_df.copy()
                    # 如果样本量大，按每200条分批预测并记录每批的路段ID
                    batch_size = 200
                    preds_pci = np.zeros(len(feature_df))
                    preds_pssi = np.zeros(len(feature_df))
                    preds_pci_std = np.zeros(len(feature_df))
                    preds_pssi_std = np.zeros(len(feature_df))
                    batch_info = []
                    for start in range(0, len(feature_df), batch_size):
                        end = min(start + batch_size, len(feature_df))
                        X_batch = feature_df.iloc[start:end]
                        # 支持 ensemble 模型对象或单模型
                        pci_model_obj = st.session_state.pso_models.get('pci')
                        pssi_model_obj = st.session_state.pso_models.get('pssi')
                        if isinstance(pci_model_obj, list):
                            arr = np.vstack([m.predict(X_batch) for m in pci_model_obj])
                            preds_pci[start:end] = np.mean(arr, axis=0)
                            preds_pci_std[start:end] = np.std(arr, axis=0)
                        else:
                            preds_pci[start:end] = pci_model_obj.predict(X_batch)
                            preds_pci_std[start:end] = 0.0

                        if isinstance(pssi_model_obj, list):
                            arr2 = np.vstack([m.predict(X_batch) for m in pssi_model_obj])
                            preds_pssi[start:end] = np.mean(arr2, axis=0)
                            preds_pssi_std[start:end] = np.std(arr2, axis=0)
                        else:
                            preds_pssi[start:end] = pssi_model_obj.predict(X_batch)
                            preds_pssi_std[start:end] = 0.0
                        # 记录该批包含的路段索引/ID（若原始 df0 含路段ID则优先使用）
                        if '路段ID' in df0.columns:
                            ids = df0['路段ID'].iloc[start:end].tolist()
                        else:
                            ids = [f'R{i}' for i in range(start, end)]
                        batch_info.append({'start': start, 'end': end, 'ids': ids})
                    base_pred['PCI'] = preds_pci
                    base_pred['PSSI'] = preds_pssi
                    base_pred['PCI_std'] = preds_pci_std
                    base_pred['PSSI_std'] = preds_pssi_std
                    # 保存批次信息到 session，供界面显示
                    st.session_state.batch_predict_info = batch_info

                    # 诊断：若模型预测几乎相同（方差接近 0），提示用户并展示特征样例
                    if not aug_enabled and (np.nanstd(preds_pci) < 1e-6 or np.nanstd(preds_pssi) < 1e-6):
                        st.warning('注意：模型预测结果方差极低，可能是特征全相同或训练数据异常。已展示特征样例以便排查。')
                        st.write('特征样例（前5行）：')
                        st.dataframe(feature_df.head(), use_container_width=True)

                    # 构造 base_df，与预测结果长度对齐（防止 X 和原始 df 长度不一致）
                    n_pred = len(base_pred)
                    # 路段ID 对齐或填充
                    if '路段ID' in df0.columns:
                        ids_series = df0['路段ID'].reset_index(drop=True)
                        if len(ids_series) >= n_pred:
                            ids = ids_series.iloc[:n_pred].tolist()
                        else:
                            ids = ids_series.tolist() + [f'R{i}' for i in range(len(ids_series), n_pred)]
                    else:
                        ids = [f'R{i}' for i in range(n_pred)]

                    # 交通量 对齐或填充默认值
                    if '交通量' in base_pred.columns:
                        traf = base_pred['交通量'].values
                    elif '交通量' in df0.columns:
                        traf_series = df0['交通量'].reset_index(drop=True)
                        if len(traf_series) >= n_pred:
                            traf = traf_series.iloc[:n_pred].values
                        else:
                            traf = np.concatenate([traf_series.values, np.full(n_pred - len(traf_series), 5000)])
                    else:
                        traf = np.full(n_pred, 5000)

                    # 路龄 对齐或填充默认值
                    if '路龄' in base_pred.columns:
                        ages = base_pred['路龄'].values
                    elif '路龄' in df0.columns:
                        age_series = df0['路龄'].reset_index(drop=True)
                        if len(age_series) >= n_pred:
                            ages = age_series.iloc[:n_pred].values
                        else:
                            ages = np.concatenate([age_series.values, np.full(n_pred - len(age_series), 5)])
                    else:
                        ages = np.full(n_pred, 5)

                    # 计算预测健康分及其不确定度（PCI 60% + PSSI 40% 权重）
                    pci_w = 0.6
                    pssi_w = 0.4
                    pred_health = pci_w * base_pred['PCI'].values + pssi_w * base_pred['PSSI'].values
                    pred_health_std = np.sqrt((pci_w * base_pred['PCI_std'].values) ** 2 + (pssi_w * base_pred['PSSI_std'].values) ** 2)

                    base_df = pd.DataFrame({
                        '路段ID': ids,
                        'PCI': base_pred['PCI'].values,
                        'PSSI': base_pred['PSSI'].values,
                        'PCI_std': base_pred['PCI_std'].values,
                        'PSSI_std': base_pred['PSSI_std'].values,
                        '预测健康分': pred_health,
                        '预测健康分_std': pred_health_std,
                        '交通量': traf,
                        '路龄': ages
                    })

                    forecast_df = forecast_pci_pssi(base_df, years=years)
                    st.session_state.processed_data = forecast_df

                    # 在界面显示批次信息（如果存在）
                    if 'batch_predict_info' in st.session_state:
                        info = st.session_state.batch_predict_info
                        st.write('每批预测信息：')
                        for b in info:
                            st.write(f"样本索引 {b['start']}:{b['end']}，路段IDs: {b['ids']}")

                    st.success('PSO-XGBoost 训练并年度预测完成')

        if st.session_state.processed_data is not None:
            forecast_df = st.session_state.processed_data
            available_years = sorted(forecast_df['年'].unique())
            sel_year = st.selectbox("选择显示年份", available_years)
            df_year = forecast_df[forecast_df['年'] == int(sel_year)]

            c1, c2 = st.columns(2)
            with c1:
                if not df_year.empty:
                    df_melt = df_year.melt(id_vars=['路段ID'], value_vars=['预测PCI','预测PSSI'], var_name='指标', value_name='值')
                    fig = px.bar(df_melt, x='路段ID', y='值', color='指标', barmode='group',
                                 title=f'各路段 {sel_year} 年预测 PCI / PSSI')
                    st.plotly_chart(fig)
                else:
                    st.info('所选年份无数据')
            with c2:
                if not df_year.empty:
                    fig2 = px.scatter(df_year, x='交通量', y='预测PCI', size='路龄', hover_name='路段ID',
                                     title=f'交通量 vs 预测PCI (年={sel_year})')
                    st.plotly_chart(fig2)
                else:
                    st.info('所选年份无数据')

# --- M3 养护决策模块 (基于专家知识库) ---
elif menu == "M3 养护决策模块 (P0)":
    st.header("🧠 智能决策生成 ")
    if st.session_state.processed_data is None:
        st.warning("请先在 M2 运行预测")
    else:
        df_all = st.session_state.processed_data.copy()

        # 如果按年展开，选择年份
        if '年' in df_all.columns:
            years_available = sorted(df_all['年'].unique())
            sel_year = st.selectbox('请选择要决策的年份', years_available)
            df = df_all[df_all['年'] == int(sel_year)].copy()
        else:
            sel_year = None
            df = df_all.copy()

        # 先确保存在预测健康分列，否则用 PCI/PSSI 加权生成
        if '预测健康分' not in df.columns:
            df['预测健康分'] = df.apply(lambda x: (float(x.get('预测PCI', x.get('PCI', 0))) * 0.6 + float(x.get('预测PSSI', x.get('PSSI', 0))) * 0.4), axis=1)
        # 如果存在预测不确定度列则保留，否则尝试从已计算的字段填充为 0
        if '预测健康分_std' not in df.columns:
            if '预测健康分_std' in df.columns:
                pass
            else:
                # 若已存在 PCI_std/PSSI_std 则合成，否则填 0
                if 'PCI_std' in df.columns or 'PSSI_std' in df.columns:
                    pci_s = df.get('PCI_std', pd.Series(0, index=df.index)).astype(float)
                    pssi_s = df.get('PSSI_std', pd.Series(0, index=df.index)).astype(float)
                    df['预测健康分_std'] = np.sqrt((0.6 * pci_s) ** 2 + (0.4 * pssi_s) ** 2)
                else:
                    df['预测健康分_std'] = 0.0

        def get_pro_decision(row):
            score = float(row.get('预测健康分', 0))
            if score >= 85:
                return "雾封层"
            elif 80 <= score < 85:
                return "同步碎石封层"
            elif 70 <= score < 80:
                return "局部铣刨重铺"
            else:
                return "结构大修"

        # 提供情景化选项：平均/悲观/乐观
        scenario = st.selectbox('选择情景', ['平均预测', '悲观预测（均值-1σ）', '乐观预测（均值+1σ）', '显示所有情景'], index=0)

        def apply_decision_with_score(score):
            if score >= 85:
                return "雾封层"
            elif 80 <= score < 85:
                return "同步碎石封层"
            elif 70 <= score < 80:
                return "局部铣刨重铺"
            else:
                return "结构大修"

        # 计算情景化的预测健康分
        if scenario == '平均预测':
            df['情景预测健康分'] = df['预测健康分']
            df['情景预测健康分_std'] = df['预测健康分_std']
            df['决策方案'] = df['情景预测健康分'].apply(apply_decision_with_score)
        elif scenario == '悲观预测（均值-1σ）':
            df['情景预测健康分'] = df['预测健康分'] - df['预测健康分_std']
            df['情景预测健康分_std'] = df['预测健康分_std']
            df['决策方案'] = df['情景预测健康分'].apply(apply_decision_with_score)
        elif scenario == '乐观预测（均值+1σ）':
            df['情景预测健康分'] = df['预测健康分'] + df['预测健康分_std']
            df['情景预测健康分_std'] = df['预测健康分_std']
            df['决策方案'] = df['情景预测健康分'].apply(apply_decision_with_score)
        else:
            # 显示所有情景：同时生成三列并合并成决策描述
            df['情景_平均'] = df['预测健康分']
            df['情景_悲观'] = df['预测健康分'] - df['预测健康分_std']
            df['情景_乐观'] = df['预测健康分'] + df['预测健康分_std']
            df['决策_平均'] = df['情景_平均'].apply(apply_decision_with_score)
            df['决策_悲观'] = df['情景_悲观'].apply(apply_decision_with_score)
            df['决策_乐观'] = df['情景_乐观'].apply(apply_decision_with_score)
            df['决策方案'] = df.apply(lambda r: f"平均:{r['决策_平均']} | 悲观:{r['决策_悲观']} | 乐观:{r['决策_乐观']}", axis=1)

        # 将专家知识详情合并到表中，便于后续导出
        def attach_expert_info(scheme):
            info = EXPERT_KNOWLEDGE.get(scheme, {})
            return info.get('参数',''), info.get('窗口期',''), info.get('效果','')

        df[['核心参数', '建议施工期', '预期效果']] = df['决策方案'].apply(lambda s: pd.Series(attach_expert_info(s)))

        # 合并回 processed_data，保留 df_all 中已有列，仅对缺失值进行填充
        merge_cols = ['决策方案','核心参数','建议施工期','预期效果','预测健康分']
        if '年' in df_all.columns:
            right_df = df[['路段ID','年'] + merge_cols].copy()
            # 使用后缀合并，然后逐列用右侧值填充左侧缺失
            merged = df_all.merge(right_df, on=['路段ID','年'], how='left', suffixes=('','_new'))
            for col in merge_cols:
                new_col = col + '_new'
                if new_col in merged.columns:
                    if col in merged.columns:
                        merged[col] = merged[col].fillna(merged[new_col])
                        merged.drop(columns=[new_col], inplace=True)
                    else:
                        merged.rename(columns={new_col: col}, inplace=True)
            df_all = merged
        else:
            right_df = df[['路段ID'] + merge_cols].copy()
            merged = df_all.merge(right_df, on=['路段ID'], how='left', suffixes=('','_new'))
            for col in merge_cols:
                new_col = col + '_new'
                if new_col in merged.columns:
                    if col in merged.columns:
                        merged[col] = merged[col].fillna(merged[new_col])
                        merged.drop(columns=[new_col], inplace=True)
                    else:
                        merged.rename(columns={new_col: col}, inplace=True)
            df_all = merged
        st.session_state.processed_data = df_all

        # --- NSGA 多目标优化层：养护效益最大化 & 成本最小化 ---
        with st.expander('🔍 NSGA 多目标优化（养护效益↑ / 成本↓）', expanded=False):
            st.write('该模块在选定路段集合上运行轻量级 NSGA-II，决策变量为每段的维护方案（离散4类）。')
            # 可选的候选路段：按预测健康分排序（低分优先）
            max_candidates = min(50, len(df))
            top_k = st.slider('选择待优化路段数量 (按健康分低→高)', 1, max_candidates, min(10, max_candidates))
            df_sorted = df.sort_values('预测健康分')
            candidates = df_sorted['路段ID'].unique().tolist()[:top_k]
            st.write(f'将对以下 {len(candidates)} 个路段进行组合优化（可在表中修改选择）')
            chosen = st.multiselect('选择要优化的路段', candidates, default=candidates)

            # 定义动作与对应的收益/成本估计（若 EXPERT_KNOWLEDGE 中含效果区间则解析）
            actions = ['雾封层','同步碎石封层','局部铣刨重铺','结构大修']
            # 基础成本估计（相对值）
            cost_map = {'雾封层':1.0, '同步碎石封层':3.0, '局部铣刨重铺':8.0, '结构大修':20.0}

            def parse_effect_gain(s):
                # 尝试解析示例文本中的 PCI 提升区间 "PCI提升5~8"，否则返回 None
                if not isinstance(s, str):
                    return None
                import re
                m = re.search(r'提升\s*(\d+)[^\d]*(?:~|\-|–)?\s*(\d+)?', s)
                if m:
                    a = float(m.group(1))
                    b = float(m.group(2)) if m.group(2) else a
                    return (a + b)/2.0
                return None

            # 为选择的路段构建收益与成本矩阵
            sel_df = df[df['路段ID'].isin(chosen)].copy()
            m = len(sel_df)
            if m == 0:
                st.info('未选择任何路段进行优化')
            else:
                benefit_mat = np.zeros((m, len(actions)))
                cost_mat = np.zeros((m, len(actions)))
                for i, (_, row) in enumerate(sel_df.iterrows()):
                    for j, a in enumerate(actions):
                        info = EXPERT_KNOWLEDGE.get(a, {})
                        gain = parse_effect_gain(info.get('效果',''))
                        if gain is None:
                            # 回退默认增益
                            default_gain = {'雾封层':6.0, '同步碎石封层':10.0, '局部铣刨重铺':17.0, '结构大修':30.0}
                            gain = default_gain.get(a, 5.0)
                        # 根据路况对收益进行微调：路龄和交通量越高，收益倾向略增
                        traffic = float(row.get('交通量', 5000))
                        age = float(row.get('路龄', 5))
                        adj = 1.0 + (traffic / 40000.0) * 0.1 + (age / 30.0) * 0.05
                        benefit_mat[i, j] = gain * adj
                        cost_mat[i, j] = cost_map.get(a, 5.0)

                st.write('示例：首行路段的动作收益/成本（前5个动作）')
                st.write(pd.DataFrame({'动作':actions, '收益示例':benefit_mat[0].round(2), '成本示例':cost_mat[0].round(2)}))

                # 运行 NSGA-II（离散动作通过四舍五入实数编码）
                run_nsga = st.button('运行 NSGA-II 优化')
                if run_nsga:
                    with st.spinner('NSGA-II 运行中...'):
                        class RoadOptProblem(Problem):
                            def __init__(self, n_var, n_obj=2):
                                super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0.0, xu=float(len(actions)-1), type_var=np.double)

                            def _evaluate(self, X, out, *args, **kwargs):
                                # X: (pop, n_var)
                                F = np.zeros((X.shape[0], 2))
                                for k in range(X.shape[0]):
                                    x = np.round(X[k]).astype(int)
                                    total_benefit = 0.0
                                    total_cost = 0.0
                                    for idx_var, act_idx in enumerate(x):
                                        if 0 <= act_idx < len(actions):
                                            total_benefit += benefit_mat[idx_var, act_idx]
                                            total_cost += cost_mat[idx_var, act_idx]
                                    # pymoo 最小化目标：为使效益最大化，使用 -total_benefit
                                    F[k,0] = -total_benefit
                                    F[k,1] = total_cost
                                out['F'] = F

                        prob = RoadOptProblem(n_var=m)
                        algorithm = NSGA2(pop_size=40)
                        res = minimize(prob, algorithm, ('n_gen', 40), seed=1, verbose=False)

                        # 提取帕累托前沿解
                        Xp = np.round(res.X).astype(int)
                        F = res.F
                        # 组装展示表格：显示每个解的总收益/成本与每段动作摘要
                        rows = []
                        for i in range(len(Xp)):
                            acts = [actions[idx] if 0<=idx<len(actions) else '未知' for idx in Xp[i]]
                            rows.append({'解号': i+1, '总收益(估计)': round(-F[i,0],2), '总成本(估计)': round(F[i,1],2), '动作摘要': ';'.join(acts)})
                        res_df = pd.DataFrame(rows)
                        st.success('NSGA-II 运行完成，展示帕累托解（示例）')
                        st.dataframe(res_df)
                        # 将首个帕累托方案应用回表格（可下载或导出）
                        if not res_df.empty:
                            pick = st.selectbox('选择要导出的方案（按行）', res_df['解号'].tolist())
                            sel_row = res_df[res_df['解号']==pick].iloc[0]
                            if st.button('将选定方案应用到 processed_data 并生成导出'): 
                                sel_idx = int(sel_row['解号']) - 1
                                chosen_actions = Xp[sel_idx]
                                # 为每个选定路段在 df_all 中写入决策方案
                                for ii, rid in enumerate(sel_df['路段ID'].values):
                                    act_name = actions[int(chosen_actions[ii])]
                                    df_all.loc[df_all['路段ID']==rid, '决策方案'] = act_name
                                st.session_state.processed_data = df_all
                                st.success('已将选定方案写入 processed_data，可在 M4 导出')

        st.write("### 自动化建议列表")
        display_cols = ['路段ID','预测健康分','预测健康分_std','决策方案','核心参数','建议施工期']
        available = [c for c in display_cols if c in df.columns or c in df_all.columns]
        # 显示合并后的 df_all 中对应列
        st.table(df_all[[c for c in display_cols if c in df_all.columns]])

# --- M4 方案生成模块 (详细专业版) ---
elif menu == "M4 方案生成模块 (P1)":
    st.header('📄 标准化方案导出')
    if st.session_state.processed_data is None or '决策方案' not in st.session_state.processed_data.columns:
        st.error("请先在 M3 生成决策")
    else:
        df_all = st.session_state.processed_data
        # 年份支持
        if '年' in df_all.columns:
            years_available = sorted(df_all['年'].unique())
            sel_year = st.selectbox('选择要导出的年份', years_available)
            df_year = df_all[df_all['年'] == int(sel_year)].copy()
        else:
            sel_year = None
            df_year = df_all.copy()

        if not df_year.empty:
            # 确保关键预测列没有缺失：若缺失则用 PCI/PSSI 计算或填 0
            def safe_health(row):
                try:
                    ph = row.get('预测健康分')
                    if ph is None or (isinstance(ph, float) and math.isnan(ph)) or pd.isna(ph):
                        pci = row.get('预测PCI', row.get('PCI', 0))
                        pssi = row.get('预测PSSI', row.get('PSSI', 0))
                        try:
                            return float(pci) * 0.6 + float(pssi) * 0.4
                        except Exception:
                            return 0.0
                    return float(ph)
                except Exception:
                    return 0.0

            df_year['预测健康分'] = df_year.apply(safe_health, axis=1)
            # 填充不确定度
            if '预测健康分_std' not in df_year.columns:
                # 合成 PCI_std/PSSI_std 或填 0
                if 'PCI_std' in df_year.columns or 'PSSI_std' in df_year.columns:
                    pci_s = df_year.get('PCI_std', pd.Series(0, index=df_year.index)).astype(float)
                    pssi_s = df_year.get('PSSI_std', pd.Series(0, index=df_year.index)).astype(float)
                    df_year['预测健康分_std'] = np.sqrt((0.6 * pci_s) ** 2 + (0.4 * pssi_s) ** 2)
                else:
                    df_year['预测健康分_std'] = 0.0
            ids = df_year['路段ID'].unique().tolist()
            selected_id = st.selectbox('选择路段', ids)
            row = df_year[df_year['路段ID'] == selected_id].iloc[0]

            scheme = row.get('决策方案', '')
            detail = EXPERT_KNOWLEDGE.get(scheme, {})

            # 回退到 processed_data 中已附带的字段（如果专家库中缺失）
            core_param = detail.get('参数') or row.get('核心参数') or '未提供核心参数'
            window = detail.get('窗口期') or row.get('建议施工期') or '未提供建议施工期'
            effect = detail.get('效果') or row.get('预期效果') or '未提供预期效果'
            scene = detail.get('场景') or row.get('适用场景') or '未提供适用场景'

            if st.button('生成详细技术方案'):
                # 在界面提示缺失信息来源
                if not detail:
                    st.warning('注意：专家知识库未包含该方案，文档中将使用 M3 中生成的字段或默认文本。')

                doc = Document()
                doc.add_heading(f'路段 {selected_id} 养护技术方案', 0)

                # 一、 路段基本概况
                doc.add_heading('一、 路段基本概况', level=1)
                p = doc.add_paragraph()
                p.add_run('根据系统预测，该路段当前预测健康评分为 ').bold = True
                ph = row.get('预测健康分', None)
                if ph is None or (isinstance(ph, float) and (math.isnan(ph))) or pd.isna(ph):
                    p.add_run('N/A').underline = True
                else:
                    try:
                        p.add_run(f"{float(ph):.2f}分").underline = True
                    except Exception:
                        p.add_run(str(ph)).underline = True
                p.add_run('。推荐方案为：')
                p.add_run(f"{scheme}").bold = True

                # 二、 核心技术要求（表格）
                doc.add_heading('二、 核心技术要求', level=1)
                table = doc.add_table(rows=1, cols=2)
                table.style = 'Table Grid'
                hdr_cells = table.rows[0].cells
                hdr_cells[0].text = '维度'
                hdr_cells[1].text = '具体要求/指标'

                items = [
                    ("适用场景", scene),
                    ("核心技术参数", core_param),
                    ("预期养护效果", effect),
                    ("建议施工期", window),
                ]
                for item, val in items:
                    display_val = val
                    if display_val is None or (isinstance(display_val, float) and math.isnan(display_val)) or pd.isna(display_val):
                        display_val = '未提供'
                    row_cells = table.add_row().cells
                    row_cells[0].text = item
                    row_cells[1].text = str(display_val)

                # 三、 施工关键控制点
                doc.add_heading('三、 施工关键控制点', level=1)
                doc.add_paragraph('1. 施工前需进行现场核查，确认病害范围与系统预测一致。')
                doc.add_paragraph(f'2. {scheme} 作业应严格控制温度及用量，确保施工环境符合规范。')
                doc.add_paragraph('3. 完工后应严格执行封道养护时间，待路面强度达标后方可开放交通。')

                buffer = BytesIO()
                doc.save(buffer)
                st.download_button(
                    label='📥 下载 Word 专业版方案',
                    data=buffer.getvalue(),
                    file_name=f'Road_{selected_id}_Technical_Plan.docx',
                    mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                )
        else:
            st.info('所选年份/数据下无可导出的路段')

# --- R1 规则层（关联规则 Apriori + PLSR） ---
elif menu == "R1 规则层 (P1)":
    st.markdown("## 🔗 R1 规则层（关联规则 + PLSR） <span class='p1-badge'>[P1]</span>", unsafe_allow_html=True)
    st.subheader("在模型预测与优化之间，使用规则挖掘与回归分析辅助决策")

    if st.session_state.processed_data is None:
        st.warning("请先在 M2 路面性能预测模块运行并生成预测数据（processed_data）")
    else:
        df = st.session_state.processed_data.copy()
        st.write("说明：Apriori 用于挖掘定性条件间的关联规则；PLSR 用于定量变量间的线性回归分析。")

        # --- Apriori 部分 ---
        st.markdown("### Apriori 关联规则挖掘（定性/二值化后）")
        apriori_cols = st.multiselect("选择用于Apriori挖掘的字段（会被二值化/编码）", options=df.columns.tolist())
        if apriori_cols:
            trans = pd.DataFrame(index=df.index)
            for c in apriori_cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    med = float(df[c].median()) if df[c].notna().any() else 0.0
                    thr = st.number_input(f"数值列 {c} 的二值化阈值 (判定为 True if > thresh)", value=med, key=f"thr_{c}")
                    trans[f"{c}_gt_{thr}"] = df[c] > float(thr)
                else:
                    vals = df[c].astype(str)
                    topk = vals.value_counts().index.tolist()[:10]
                    sel = st.multiselect(f"选择 {c} 的类别用于编码（多选）", options=topk, key=f"sel_{c}")
                    for sv in sel:
                        trans[f"{c}={sv}"] = vals == sv

            if not trans.empty:
                min_sup = st.slider("最小支持度 (support)", 0.01, 1.0, 0.1)
                min_conf = st.slider("最小置信度 (confidence)", 0.01, 1.0, 0.6)
                try:
                    freq = apriori(trans.fillna(False).astype(bool), min_support=min_sup, use_colnames=True)
                    st.write("频繁项集（support 降序）：")
                    st.dataframe(freq.sort_values('support', ascending=False).head(100))
                    rules = association_rules(freq, metric='confidence', min_threshold=min_conf)
                    if not rules.empty:
                        rules = rules.sort_values('lift', ascending=False)
                        st.write("发现关联规则（按 lift 排序）：")
                        st.dataframe(rules[['antecedents','consequents','support','confidence','lift']].head(200))
                    else:
                        st.info('未找到满足阈值的关联规则')
                except Exception as e:
                    st.error(f"Apriori 挖掘失败: {e}")

        # --- PLSR 部分 ---
        st.markdown("### PLSR 回归分析（定量变量）")
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        st.write("可用数值字段：", numeric_cols)
        pls_x = st.multiselect("选择自变量 X（数值列）", options=numeric_cols, default=numeric_cols[:min(5,len(numeric_cols))])
        pls_y = st.selectbox("选择因变量 y（数值列）", options=numeric_cols, index=0)
        max_comp = min(10, max(1, len(pls_x)))
        n_comp = st.slider("PLSR 主成分数 (n_components)", 1, max_comp, min(2, max_comp))
        if st.button("运行 PLSR 分析"):
            if not pls_x or pls_y is None:
                st.error("请先选择 X 与 y 字段")
            else:
                X = df[pls_x].fillna(0).values
                y = df[pls_y].fillna(0).values
                try:
                    scaler = StandardScaler()
                    Xs = scaler.fit_transform(X)
                    pls = PLSRegression(n_components=n_comp)
                    pls.fit(Xs, y)
                    y_pred = pls.predict(Xs).ravel()
                    from sklearn.metrics import r2_score
                    r2 = r2_score(y, y_pred)
                    st.write(f"PLSR 训练集 R^2: {r2:.4f}")
                    coef = pd.Series(pls.coef_.ravel(), index=pls_x)
                    st.write("PLSR 回归系数：")
                    st.dataframe(coef)
                    st.write("预测值 vs 真实值（前50）")
                    st.dataframe(pd.DataFrame({'y_true': y, 'y_pred': y_pred}).head(50))
                except Exception as e:
                    st.error(f"PLSR 运行失败: {e}")

# --- 数据检查与分析工具 ---
def analyze_data(feature_df, y_pci, y_pssi):
    """
    分析特征和目标值的分布，打印统计信息。
    """
    st.write("### 数据分布分析")

    # 分析特征分布
    st.write("#### 特征分布统计")
    st.dataframe(feature_df.describe())

    # 分析目标值分布
    st.write("#### PCI 分布统计")
    st.dataframe(y_pci.describe())

    st.write("#### PSSI 分布统计")
    st.dataframe(y_pssi.describe())

    # 检查缺失值
    st.write("#### 缺失值统计")
    missing_values = feature_df.isnull().sum()
    st.dataframe(missing_values[missing_values > 0])

# --- 在 PSO-XGBoost 训练前调用数据分析 ---
if st.button("🔍 分析数据分布"):
    if st.session_state.raw_data is None:
        st.error("请先上传或生成数据！")
    else:
        df0 = st.session_state.raw_data.copy()
        if set(['mean_flat','mean_jump','mean_rut','mean_struct']).issubset(df0.columns):
            feature_df = df0[['mean_flat','mean_jump','mean_rut','mean_struct','交通量','路龄']].copy()
        else:
            feature_df = _build_features_from_measurements(df0)

        if 'PCI' in df0.columns and 'PSSI' in df0.columns:
            y_pci = df0['PCI']
            y_pssi = df0['PSSI']
        else:
            tmp = compute_pci_pssi_from_measurements(df0, pd.DataFrame())
            y_pci = tmp['PCI']
            y_pssi = tmp['PSSI']

        analyze_data(feature_df, y_pci, y_pssi)
