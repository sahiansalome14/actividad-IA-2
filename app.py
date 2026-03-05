import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, learning_curve
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Classifier Suite",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
    .main { background-color: #0f0a0a; }
    .stApp { background-color: #0f0a0a; }
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #c9a96e !important; }
    .model-header {
        background: linear-gradient(90deg, #1a0a0a, #2d1515);
        border-left: 4px solid #c9a96e;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1.2rem;
        margin: 1rem 0;
    }
    div[data-testid="stMetric"] {
        background: #1a1010;
        border: 1px solid #c9a96e33;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    div[data-testid="stMetric"] label { color: #888 !important; }
    div[data-testid="stMetric"] div { color: #c9a96e !important; }
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="class")
    return X, y, wine

X, y, wine = load_data()
CLASSES = wine.target_names
COLORS = ["#c9a96e", "#e07b7b", "#7bb8e0"]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍷 Wine Classifier Suite")
    st.markdown("---")

    st.markdown("### 🌳 Decision Tree")
    dt_max_depth  = st.slider("Max Depth", 1, 15, 4)
    dt_criterion  = st.selectbox("Criterion", ["gini", "entropy", "log_loss"])
    dt_min_split  = st.slider("Min Samples Split", 2, 20, 2)

    st.markdown("---")
    st.markdown("### 🎒 Bagging")
    bag_n         = st.slider("N Estimators", 5, 200, 50)
    bag_samples   = st.slider("Max Samples (fracción)", 0.3, 1.0, 0.8)
    bag_features  = st.slider("Max Features (fracción)", 0.3, 1.0, 1.0)
    bag_boot_feat = st.checkbox("Bootstrap Features", value=False)

    st.markdown("---")
    st.markdown("### 🚀 Boosting")
    boost_type    = st.radio("Algoritmo", ["AdaBoost", "Gradient Boosting"])
    boost_n       = st.slider("N Estimators", 10, 300, 100)
    boost_lr      = st.slider("Learning Rate", 0.01, 2.0, 0.1, step=0.01)
    gb_depth      = st.slider("Tree Depth (solo GB)", 1, 8, 3)
    gb_sub        = st.slider("Subsample (solo GB)", 0.3, 1.0, 1.0)

    st.markdown("---")
    st.markdown("### 🔁 Validación Cruzada")
    cv_k          = st.slider("Folds (k)", 3, 10, 5)
    cv_metrics    = st.multiselect(
        "Métricas",
        ["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        default=["accuracy", "f1_macro"]
    )

    run_btn = st.button("▶ Ejecutar Análisis", use_container_width=True, type="primary")

# ── MODEL FACTORY ─────────────────────────────────────────────────────────────
def build_models():
    base_dt = DecisionTreeClassifier(max_depth=dt_max_depth, random_state=42)
    dt  = DecisionTreeClassifier(max_depth=dt_max_depth, criterion=dt_criterion,
                                  min_samples_split=dt_min_split, random_state=42)
    bag = BaggingClassifier(estimator=base_dt, n_estimators=bag_n,
                             max_samples=bag_samples, max_features=bag_features,
                             bootstrap=True, bootstrap_features=bag_boot_feat,
                             random_state=42, n_jobs=-1)
    if boost_type == "AdaBoost":
        import sklearn
        ada_kwargs = dict(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
            n_estimators=boost_n,
            learning_rate=boost_lr,
            random_state=42
        )
        # algorithm="SAMME" was removed in sklearn 1.6+
        sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if sk_version < (1, 6):
            ada_kwargs["algorithm"] = "SAMME"
        bst = AdaBoostClassifier(**ada_kwargs)
    else:
        bst = GradientBoostingClassifier(n_estimators=boost_n, learning_rate=boost_lr,
                                          max_depth=gb_depth, subsample=gb_sub, random_state=42)
    return {"Decision Tree": dt, "Bagging": bag, boost_type: bst}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def set_dark_axes(*axes):
    for ax in axes:
        ax.set_facecolor("#1a1010")
        ax.tick_params(colors="#777")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

def plot_roc(model, X_arr, y_arr, ax, title, color):
    y_bin = label_binarize(y_arr, classes=[0, 1, 2])
    skf   = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=42)
    fpr_g = np.linspace(0, 1, 200)
    tprs  = [[] for _ in range(3)]

    for tr, te in skf.split(X_arr, y_arr):
        model.fit(X_arr[tr], y_arr[tr])
        p = model.predict_proba(X_arr[te]) if hasattr(model, "predict_proba") else \
            label_binarize(model.predict(X_arr[te]), classes=[0,1,2])
        for c in range(3):
            fpr, tpr, _ = roc_curve(y_bin[te, c], p[:, c])
            tprs[c].append(np.interp(fpr_g, fpr, tpr))

    cls_colors = ["#c9a96e", "#e07b7b", "#7bb8e0"]
    for c in range(3):
        mt = np.mean(tprs[c], axis=0)
        ma = auc(fpr_g, mt)
        ax.plot(fpr_g, mt, color=cls_colors[c], lw=2,
                label=f"{CLASSES[c]} (AUC={ma:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
    ax.set(xlim=[0,1], ylim=[0,1.02], xlabel="FPR", ylabel="TPR")
    ax.set_title(f"ROC — {title}", color="#c9a96e", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    set_dark_axes(ax)

def plot_cm(model, X_arr, y_arr, ax, title):
    skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=42)
    cm  = np.zeros((3,3), dtype=int)
    for tr, te in skf.split(X_arr, y_arr):
        model.fit(X_arr[tr], y_arr[tr])
        cm += confusion_matrix(y_arr[te], model.predict(X_arr[te]))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.4, linecolor="#222", ax=ax,
                cbar_kws={"shrink": 0.75})
    ax.set_title(f"Conf. Matrix — {title}", color="#c9a96e", fontsize=10)
    ax.set_xlabel("Predicted", color="#aaa", fontsize=8)
    ax.set_ylabel("True", color="#aaa", fontsize=8)
    ax.tick_params(colors="#aaa")

def plot_lc(model, X_arr, y_arr, ax, title, color):
    sizes, tr_s, va_s = learning_curve(
        model, X_arr, y_arr,
        cv=StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=42),
        train_sizes=np.linspace(0.1,1,10), scoring="accuracy", n_jobs=-1)
    tr_m, tr_sd = tr_s.mean(1), tr_s.std(1)
    va_m, va_sd = va_s.mean(1), va_s.std(1)
    ax.plot(sizes, tr_m, "o-", color=color, lw=2, label="Train")
    ax.fill_between(sizes, tr_m-tr_sd, tr_m+tr_sd, alpha=0.15, color=color)
    ax.plot(sizes, va_m, "s--", color="#e07b7b", lw=2, label="Validation")
    ax.fill_between(sizes, va_m-va_sd, va_m+va_sd, alpha=0.15, color="#e07b7b")
    ax.set_title(f"Learning Curve — {title}", color="#c9a96e", fontsize=10)
    ax.set_xlabel("Samples", color="#aaa", fontsize=8)
    ax.set_ylabel("Accuracy", color="#aaa", fontsize=8)
    ax.legend(fontsize=7)
    set_dark_axes(ax)

# ── MAIN UI ───────────────────────────────────────────────────────────────────
st.markdown("# 🍷 Wine Classification Suite")
st.markdown("**Árbol de Decisión · Bagging (Bootstrap Aggregating) · Boosting** — validación cruzada y métricas completas")
st.markdown("---")

tab_data, tab_models, tab_compare = st.tabs(["📊 Dataset", "🔬 Modelos", "⚖️ Comparación"])

# ── TAB: DATASET ──────────────────────────────────────────────────────────────
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Muestras", len(X))
    c2.metric("Características", X.shape[1])
    c3.metric("Clases", len(CLASSES))
    c4.metric("Balance mín/máx", f"{y.value_counts().min()}/{y.value_counts().max()}")

    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### Distribución de Clases")
        fig, ax = plt.subplots(figsize=(5,3), facecolor="#0f0a0a")
        set_dark_axes(ax)
        counts = y.value_counts().sort_index()
        bars = ax.bar(CLASSES, counts.values, color=COLORS, edgecolor="#333")
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, str(v),
                    ha="center", color="#c9a96e", fontsize=10)
        ax.set_ylabel("Count", color="#aaa")
        st.pyplot(fig, use_container_width=True)
    with cb:
        st.markdown("#### Correlación (top 8 features)")
        fig, ax = plt.subplots(figsize=(5,3.5), facecolor="#0f0a0a")
        corr = X.iloc[:,:8].corr()
        sns.heatmap(corr, ax=ax, cmap="RdYlGn", center=0,
                    linewidths=0.3, linecolor="#222",
                    mask=np.triu(np.ones_like(corr,bool)),
                    cbar_kws={"shrink":0.7}, annot=False)
        ax.tick_params(colors="#aaa", labelsize=7)
        st.pyplot(fig, use_container_width=True)

    st.markdown("#### Preview del Dataset")
    df_prev = X.copy(); df_prev["class"] = y.map(dict(enumerate(CLASSES)))
    st.dataframe(df_prev.head(10), use_container_width=True)

# ── RUN ───────────────────────────────────────────────────────────────────────
if "ready" not in st.session_state:
    st.session_state.ready = False

if run_btn:
    with st.spinner("⏳ Entrenando y evaluando modelos..."):
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X.values)
        y_arr   = y.values
        models  = build_models()
        skf     = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=42)
        metrics = cv_metrics if cv_metrics else ["accuracy"]

        cv_res = {}
        for nm, m in models.items():
            res = cross_validate(m, X_sc, y_arr, cv=skf, scoring=metrics, n_jobs=-1)
            cv_res[nm] = {k: (np.mean(v), np.std(v))
                          for k, v in res.items() if k.startswith("test_")}

        st.session_state.update(
            ready=True, models=models, cv_res=cv_res,
            X_sc=X_sc, y_arr=y_arr
        )
    st.success("✅ Listo")

# ── TAB: MODELS ───────────────────────────────────────────────────────────────
with tab_models:
    if not st.session_state.ready:
        st.info("⬅ Ajusta parámetros y presiona **▶ Ejecutar Análisis**")
    else:
        mcolors = {"Decision Tree":"#c9a96e","Bagging":"#7bb8e0",
                   "AdaBoost":"#e07b7b","Gradient Boosting":"#e07b7b"}
        icons   = {"Decision Tree":"🌳","Bagging":"🎒","AdaBoost":"🚀","Gradient Boosting":"🚀"}
        for nm, model in st.session_state.models.items():
            col = mcolors.get(nm,"#c9a96e")
            icon= icons.get(nm,"🔬")
            st.markdown(f'<div class="model-header"><h3 style="margin:0;color:{col}">{icon} {nm}</h3></div>',
                        unsafe_allow_html=True)

            # CV metric cards
            cv_data = st.session_state.cv_res[nm]
            cols = st.columns(len(cv_data))
            for c, (k,(mu,sd)) in zip(cols, cv_data.items()):
                label = k.replace("test_","").replace("_"," ").title()
                c.metric(label, f"{mu:.4f}", f"±{sd:.4f}")

            # Three-panel chart
            fig = plt.figure(figsize=(15,4), facecolor="#0f0a0a")
            ax1,ax2,ax3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)
            set_dark_axes(ax1,ax2,ax3)
            plot_roc(model, st.session_state.X_sc, st.session_state.y_arr, ax1, nm, col)
            plot_cm(model,  st.session_state.X_sc, st.session_state.y_arr, ax2, nm)
            plot_lc(model,  st.session_state.X_sc, st.session_state.y_arr, ax3, nm, col)
            plt.tight_layout(pad=2)
            st.pyplot(fig, use_container_width=True)

            # Feature importance
            model.fit(st.session_state.X_sc, st.session_state.y_arr)
            if hasattr(model, "feature_importances_"):
                with st.expander(f"📈 Importancia de Features — {nm}"):
                    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    fig2, ax = plt.subplots(figsize=(10,3), facecolor="#0f0a0a")
                    set_dark_axes(ax)
                    ax.barh(fi.index[:10], fi.values[:10], color=col, edgecolor="#333", alpha=0.85)
                    ax.set_xlabel("Importancia", color="#aaa")
                    ax.invert_yaxis()
                    st.pyplot(fig2, use_container_width=True)

            # Tree visualization
            if nm == "Decision Tree":
                with st.expander("🌲 Visualizar Árbol"):
                    fig3, ax = plt.subplots(figsize=(14,6), facecolor="#0f0a0a")
                    plot_tree(model, feature_names=list(X.columns),
                              class_names=list(CLASSES), filled=True, rounded=True,
                              max_depth=min(3, dt_max_depth), ax=ax, fontsize=7)
                    st.pyplot(fig3, use_container_width=True)

            st.markdown("---")

# ── TAB: COMPARISON ───────────────────────────────────────────────────────────
with tab_compare:
    if not st.session_state.ready:
        st.info("⬅ Ajusta parámetros y presiona **▶ Ejecutar Análisis**")
    else:
        st.markdown("### 📊 Tabla Comparativa")
        rows = []
        for nm, cvd in st.session_state.cv_res.items():
            row = {"Modelo": nm}
            for k,(mu,sd) in cvd.items():
                row[k.replace("test_","").replace("_"," ").title()] = f"{mu:.4f} ± {sd:.4f}"
            rows.append(row)
        df_cmp = pd.DataFrame(rows).set_index("Modelo")
        st.dataframe(df_cmp, use_container_width=True)

        # Bar chart per metric
        metric_data = {}
        for nm, cvd in st.session_state.cv_res.items():
            for k,(mu,_) in cvd.items():
                lbl = k.replace("test_","")
                metric_data.setdefault(lbl,{})[nm] = mu

        if metric_data:
            st.markdown("### 📈 Comparación Visual")
            n_metrics = len(metric_data)
            fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4), facecolor="#0f0a0a")
            if n_metrics == 1: axes = [axes]
            bar_colors = ["#c9a96e","#7bb8e0","#e07b7b"]
            mnames = list(st.session_state.models.keys())
            for ax, (metric, scores) in zip(axes, metric_data.items()):
                set_dark_axes(ax)
                vals = [scores.get(m,0) for m in mnames]
                bars = ax.bar(mnames, vals, color=bar_colors[:len(mnames)],
                              edgecolor="#333", alpha=0.88)
                for b, v in zip(bars, vals):
                    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                            f"{v:.4f}", ha="center", va="bottom", color="#ccc", fontsize=8)
                ax.set_title(metric.replace("_"," ").title(), color="#c9a96e", fontsize=11)
                ax.set_ylim(max(0, min(vals)-0.05), 1.0)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=12, ha="right")
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)

        # ROC side by side
        st.markdown("### 🔵 Curvas ROC — todos los modelos")
        fig, axes = plt.subplots(1,3, figsize=(15,4), facecolor="#0f0a0a")
        mcolors = {"Decision Tree":"#c9a96e","Bagging":"#7bb8e0",
                   "AdaBoost":"#e07b7b","Gradient Boosting":"#e07b7b"}
        for ax,(nm,m) in zip(axes, st.session_state.models.items()):
            set_dark_axes(ax)
            plot_roc(m, st.session_state.X_sc, st.session_state.y_arr, ax, nm,
                     mcolors.get(nm,"#c9a96e"))
        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)

        # Confusion Matrices side by side
        st.markdown("### 🟥 Matrices de Confusión — todos los modelos")
        fig, axes = plt.subplots(1,3, figsize=(15,4), facecolor="#0f0a0a")
        for ax,(nm,m) in zip(axes, st.session_state.models.items()):
            set_dark_axes(ax)
            plot_cm(m, st.session_state.X_sc, st.session_state.y_arr, ax, nm)
        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)

        # Winner
        if "accuracy" in metric_data:
            best = max(metric_data["accuracy"], key=lambda k: metric_data["accuracy"][k])
            bval = metric_data["accuracy"][best]
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a2a1a,#2a3a2a);
                        border:2px solid #4caf50;border-radius:12px;
                        padding:1.5rem;text-align:center;margin-top:1rem;">
              <div style="font-size:2.5rem">🏆</div>
              <div style="font-family:'Playfair Display',serif;color:#c9a96e;font-size:1.8rem">{best}</div>
              <div style="color:#4caf50;font-size:1.2rem;font-weight:600">Accuracy CV: {bval:.4f}</div>
              <div style="color:#888;font-size:0.8rem;margin-top:0.3rem">
                Mejor desempeño promedio en {cv_k}-fold cross-validation
              </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#555;font-size:0.8rem">Wine Classification Suite · sklearn · Streamlit</div>',
            unsafe_allow_html=True)
