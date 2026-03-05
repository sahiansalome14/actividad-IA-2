from flask import Flask, render_template_string, jsonify, request
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, RandomForestClassifier)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.multiclass import OneVsRestClassifier
import base64
import io

app = Flask(__name__)

# ── HTML Template ────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ML Wine Classifier Lab</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --card: #1a1a26;
    --border: #2a2a3d;
    --accent: #7c3aed;
    --accent2: #06b6d4;
    --accent3: #f59e0b;
    --accent4: #10b981;
    --text: #e2e8f0;
    --muted: #64748b;
    --danger: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Syne', sans-serif; min-height: 100vh; }

  /* HEADER */
  header {
    background: linear-gradient(135deg, #12121a 0%, #1e1040 50%, #0d1a2d 100%);
    border-bottom: 1px solid var(--border);
    padding: 2rem 2rem 1.5rem;
    position: relative; overflow: hidden;
  }
  header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(124,58,237,.15) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(6,182,212,.1) 0%, transparent 50%);
  }
  header .inner { position: relative; max-width: 1200px; margin: 0 auto; }
  header h1 { font-size: clamp(1.8rem,4vw,3rem); font-weight: 800; letter-spacing: -.02em; }
  header h1 span { color: var(--accent2); }
  header p { color: var(--muted); font-size: .95rem; margin-top: .4rem; font-family: 'Space Mono', monospace; }

  /* MAIN */
  main { max-width: 1200px; margin: 0 auto; padding: 2rem; }

  /* CONFIG PANEL */
  .config-panel {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.5rem; margin-bottom: 2rem;
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.5rem;
  }
  .config-group label { display: block; color: var(--muted); font-size: .8rem;
    font-family: 'Space Mono', monospace; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .5rem; }
  .config-group input[type=range] { width: 100%; accent-color: var(--accent); }
  .range-val { color: var(--accent2); font-family: 'Space Mono', monospace; font-size: .9rem; font-weight: 700; }
  .config-group select {
    width: 100%; background: var(--surface); border: 1px solid var(--border);
    color: var(--text); padding: .5rem .75rem; border-radius: 6px; font-family: 'Syne', sans-serif;
  }
  .config-group .checkbox-group { display: flex; flex-direction: column; gap: .4rem; }
  .config-group .checkbox-group label {
    display: flex; align-items: center; gap: .5rem; color: var(--text);
    font-size: .85rem; text-transform: none; letter-spacing: 0; cursor: pointer;
  }
  .config-group .checkbox-group input[type=checkbox] { accent-color: var(--accent); width: 14px; height: 14px; }

  .run-btn {
    grid-column: 1 / -1;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff; border: none; border-radius: 8px;
    padding: .8rem 2rem; font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 700; cursor: pointer;
    transition: opacity .2s, transform .1s; letter-spacing: .03em;
  }
  .run-btn:hover { opacity: .9; }
  .run-btn:active { transform: scale(.98); }
  .run-btn:disabled { opacity: .5; cursor: not-allowed; }

  /* STATUS */
  #status {
    font-family: 'Space Mono', monospace; font-size: .85rem;
    color: var(--accent3); padding: .6rem 1rem;
    background: rgba(245,158,11,.07); border: 1px solid rgba(245,158,11,.2);
    border-radius: 6px; margin-bottom: 1.5rem; display: none;
  }

  /* RESULTS */
  #results { display: none; }

  /* MODEL CARDS */
  .models-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.2rem; margin-bottom: 2rem; }
  .model-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.2rem; position: relative; overflow: hidden;
    transition: border-color .2s;
  }
  .model-card:hover { border-color: var(--accent); }
  .model-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  }
  .model-card.dt::before { background: var(--accent); }
  .model-card.bag::before { background: var(--accent2); }
  .model-card.boost::before { background: var(--accent3); }
  .model-card.gb::before { background: var(--accent4); }
  .model-card h3 { font-size: .95rem; font-weight: 600; margin-bottom: .8rem; }
  .model-card .tag {
    font-family: 'Space Mono', monospace; font-size: .7rem;
    padding: .2rem .5rem; border-radius: 4px; margin-bottom: .8rem; display: inline-block;
  }
  .dt .tag { background: rgba(124,58,237,.15); color: var(--accent); }
  .bag .tag { background: rgba(6,182,212,.15); color: var(--accent2); }
  .boost .tag { background: rgba(245,158,11,.15); color: var(--accent3); }
  .gb .tag { background: rgba(16,185,129,.15); color: var(--accent4); }

  .metric-row { display: flex; justify-content: space-between; align-items: center;
    padding: .3rem 0; border-bottom: 1px solid rgba(255,255,255,.04); font-size: .85rem; }
  .metric-row:last-child { border-bottom: none; }
  .metric-row .name { color: var(--muted); font-family: 'Space Mono', monospace; font-size: .75rem; }
  .metric-row .val { font-weight: 700; font-family: 'Space Mono', monospace; }
  .bar-track { background: var(--surface); border-radius: 99px; height: 4px; flex: 1; margin: 0 .6rem; }
  .bar-fill { height: 100%; border-radius: 99px; transition: width .8s cubic-bezier(.4,0,.2,1); }
  .dt .bar-fill { background: var(--accent); }
  .bag .bar-fill { background: var(--accent2); }
  .boost .bar-fill { background: var(--accent3); }
  .gb .bar-fill { background: var(--accent4); }

  /* CV Section */
  .section-title {
    font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem;
    display: flex; align-items: center; gap: .5rem;
  }
  .section-title::before {
    content: ''; width: 4px; height: 1.1rem; border-radius: 2px;
    background: linear-gradient(var(--accent), var(--accent2));
  }

  /* Charts grid */
  .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
  .chart-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem;
  }
  .chart-card h4 { font-size: .85rem; color: var(--muted); font-family: 'Space Mono', monospace;
    text-transform: uppercase; letter-spacing: .06em; margin-bottom: 1rem; }
  .chart-card canvas { max-height: 280px; }

  /* Confusion matrix */
  .cm-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
  .cm-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; }
  .cm-card h4 { font-size: .85rem; color: var(--muted); font-family: 'Space Mono', monospace;
    text-transform: uppercase; letter-spacing: .06em; margin-bottom: 1rem; }
  .cm-table { width: 100%; border-collapse: collapse; font-family: 'Space Mono', monospace; font-size: .75rem; }
  .cm-table td, .cm-table th { text-align: center; padding: .4rem; border: 1px solid var(--border); }
  .cm-table th { background: var(--surface); color: var(--muted); }
  .cm-cell-high { background: rgba(124,58,237,.4); color: #fff; font-weight: 700; }
  .cm-cell-med  { background: rgba(124,58,237,.2); }
  .cm-cell-low  { background: transparent; color: var(--muted); }

  /* Report table */
  .report-section { margin-bottom: 2rem; }
  .report-table { width: 100%; border-collapse: collapse; font-family: 'Space Mono', monospace; font-size: .78rem; }
  .report-table th { background: var(--surface); color: var(--muted); text-align: left;
    padding: .5rem .75rem; border-bottom: 1px solid var(--border); }
  .report-table td { padding: .5rem .75rem; border-bottom: 1px solid rgba(255,255,255,.04); }
  .report-table tr:hover td { background: rgba(255,255,255,.02); }
  .badge { display: inline-block; padding: .15rem .4rem; border-radius: 4px; font-size: .72rem; }
  .badge-dt    { background: rgba(124,58,237,.15); color: var(--accent); }
  .badge-bag   { background: rgba(6,182,212,.15);  color: var(--accent2); }
  .badge-boost { background: rgba(245,158,11,.15); color: var(--accent3); }
  .badge-gb    { background: rgba(16,185,129,.15); color: var(--accent4); }

  /* Responsive */
  @media(max-width:600px) { main { padding: 1rem; } }
</style>
</head>
<body>
<header>
  <div class="inner">
    <h1>🍷 ML <span>Wine</span> Lab</h1>
    <p>// Decision Tree · Bagging · Boosting · Gradient Boosting — comparativa completa</p>
  </div>
</header>
<main>

  <!-- CONFIG -->
  <div class="config-panel">
    <div class="config-group">
      <label>Profundidad máx. árbol: <span class="range-val" id="depthVal">3</span></label>
      <input type="range" id="maxDepth" min="1" max="10" value="3"
             oninput="document.getElementById('depthVal').textContent=this.value">
    </div>
    <div class="config-group">
      <label>CV Folds: <span class="range-val" id="foldVal">5</span></label>
      <input type="range" id="cvFolds" min="2" max="10" value="5"
             oninput="document.getElementById('foldVal').textContent=this.value">
    </div>
    <div class="config-group">
      <label>Estimadores (ensemble): <span class="range-val" id="estVal">50</span></label>
      <input type="range" id="nEst" min="10" max="200" step="10" value="50"
             oninput="document.getElementById('estVal').textContent=this.value">
    </div>
    <div class="config-group">
      <label>Modelos a comparar</label>
      <div class="checkbox-group">
        <label><input type="checkbox" id="chkDT"    checked> 🌳 Decision Tree</label>
        <label><input type="checkbox" id="chkBag"   checked> 👜 Bagging</label>
        <label><input type="checkbox" id="chkBoost" checked> 🚀 AdaBoost</label>
        <label><input type="checkbox" id="chkGB"    checked> 📈 Gradient Boosting</label>
      </div>
    </div>
    <button class="run-btn" id="runBtn" onclick="runExperiment()">▶ Ejecutar experimento</button>
  </div>

  <div id="status"></div>
  <div id="results">
    <div class="section-title">Resumen por modelo</div>
    <div class="models-grid" id="modelCards"></div>

    <div class="section-title">Validación cruzada (CV Scores)</div>
    <div class="charts-grid">
      <div class="chart-card"><h4>Accuracy por fold (CV)</h4><canvas id="cvChart"></canvas></div>
      <div class="chart-card"><h4>Distribución de métricas</h4><canvas id="metricsRadar"></canvas></div>
      <div class="chart-card"><h4>Curva ROC (test set, OvR)</h4><canvas id="rocChart"></canvas></div>
      <div class="chart-card"><h4>Importancia de features (DT)</h4><canvas id="featChart"></canvas></div>
    </div>

    <div class="section-title">Matrices de confusión</div>
    <div class="cm-grid" id="cmGrid"></div>

    <div class="section-title report-section">Reporte detallado (test set)</div>
    <div class="report-section">
      <table class="report-table" id="reportTable">
        <thead>
          <tr>
            <th>Modelo</th><th>Clase</th><th>Precision</th>
            <th>Recall</th><th>F1</th><th>Support</th>
          </tr>
        </thead>
        <tbody id="reportBody"></tbody>
      </table>
    </div>
  </div>
</main>

<script>
let cvChart, radarChart, rocChart, featChart;

function setStatus(msg) {
  const el = document.getElementById('status');
  el.style.display = msg ? 'block' : 'none';
  el.textContent = '⏳ ' + msg;
}

async function runExperiment() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  setStatus('Entrenando modelos y calculando métricas...');

  const payload = {
    max_depth: parseInt(document.getElementById('maxDepth').value),
    cv_folds:  parseInt(document.getElementById('cvFolds').value),
    n_est:     parseInt(document.getElementById('nEst').value),
    models: {
      dt:    document.getElementById('chkDT').checked,
      bag:   document.getElementById('chkBag').checked,
      boost: document.getElementById('chkBoost').checked,
      gb:    document.getElementById('chkGB').checked,
    }
  };

  try {
    const res = await fetch('/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResults(data);
    document.getElementById('results').style.display = 'block';
    setStatus('');
  } catch(e) {
    setStatus('Error: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

// ── PALETTE ────────────────────────────────────────────────────────────────
const COLORS = {
  dt:    { border:'#7c3aed', bg:'rgba(124,58,237,.15)',   cls:'dt'    },
  bag:   { border:'#06b6d4', bg:'rgba(6,182,212,.15)',    cls:'bag'   },
  boost: { border:'#f59e0b', bg:'rgba(245,158,11,.15)',   cls:'boost' },
  gb:    { border:'#10b981', bg:'rgba(16,185,129,.15)',   cls:'gb'    },
};
const LABELS = {
  dt:'Decision Tree', bag:'Bagging', boost:'AdaBoost', gb:'Gradient Boosting'
};

function renderResults(data) {
  renderCards(data);
  renderCV(data);
  renderRadar(data);
  renderROC(data);
  renderFeatures(data);
  renderCM(data);
  renderReport(data);
}

// ── MODEL CARDS ─────────────────────────────────────────────────────────────
function renderCards(data) {
  const grid = document.getElementById('modelCards');
  grid.innerHTML = '';
  const mkeys = Object.keys(data.results);
  mkeys.forEach(mk => {
    const r = data.results[mk];
    const c = COLORS[mk];
    const metricDefs = [
      {k:'accuracy',  label:'Accuracy'},
      {k:'precision', label:'Precision'},
      {k:'recall',    label:'Recall'},
      {k:'f1',        label:'F1-Score'},
      {k:'roc_auc',   label:'ROC-AUC'},
      {k:'cv_mean',   label:'CV Mean'},
      {k:'cv_std',    label:'CV Std'},
    ];
    const rows = metricDefs.map(m => {
      const v = r[m.k];
      const pct = Math.min(100, v*100);
      return `<div class="metric-row">
        <span class="name">${m.label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        <span class="val">${v.toFixed(4)}</span>
      </div>`;
    }).join('');
    grid.innerHTML += `
      <div class="model-card ${c.cls}">
        <span class="tag">${mk.toUpperCase()}</span>
        <h3>${LABELS[mk]}</h3>
        ${rows}
      </div>`;
  });
}

// ── CV CHART ─────────────────────────────────────────────────────────────────
function renderCV(data) {
  if (cvChart) cvChart.destroy();
  const mkeys = Object.keys(data.results);
  const folds = data.results[mkeys[0]].cv_scores.length;
  const labels = Array.from({length:folds}, (_,i)=>`Fold ${i+1}`);
  const datasets = mkeys.map(mk => ({
    label: LABELS[mk],
    data: data.results[mk].cv_scores,
    borderColor: COLORS[mk].border,
    backgroundColor: COLORS[mk].bg,
    tension: .3, fill: false, pointRadius: 5,
  }));
  cvChart = new Chart(document.getElementById('cvChart'), {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive:true, maintainAspectRatio:true,
      plugins:{ legend:{ labels:{ color:'#94a3b8', font:{family:'Space Mono',size:10} } } },
      scales:{
        x:{ ticks:{color:'#64748b'}, grid:{color:'rgba(255,255,255,.05)'} },
        y:{ ticks:{color:'#64748b'}, grid:{color:'rgba(255,255,255,.05)'}, min:.5, max:1 }
      }
    }
  });
}

// ── RADAR ─────────────────────────────────────────────────────────────────────
function renderRadar(data) {
  if (radarChart) radarChart.destroy();
  const mkeys = Object.keys(data.results);
  const axes = ['accuracy','precision','recall','f1','roc_auc'];
  const axLabels = ['Accuracy','Precision','Recall','F1','ROC-AUC'];
  const datasets = mkeys.map(mk => ({
    label: LABELS[mk],
    data: axes.map(a => data.results[mk][a]),
    borderColor: COLORS[mk].border,
    backgroundColor: COLORS[mk].bg,
    pointBackgroundColor: COLORS[mk].border,
  }));
  radarChart = new Chart(document.getElementById('metricsRadar'), {
    type: 'radar',
    data: { labels: axLabels, datasets },
    options: {
      responsive:true, maintainAspectRatio:true,
      plugins:{ legend:{ labels:{ color:'#94a3b8', font:{family:'Space Mono',size:10} } } },
      scales:{ r:{ min:.5, max:1, ticks:{color:'#64748b',backdropColor:'transparent'},
        grid:{color:'rgba(255,255,255,.08)'}, pointLabels:{color:'#94a3b8',font:{size:10}} } }
    }
  });
}

// ── ROC CHART ─────────────────────────────────────────────────────────────────
function renderROC(data) {
  if (rocChart) rocChart.destroy();
  const mkeys = Object.keys(data.results);
  const datasets = [];
  mkeys.forEach(mk => {
    if (!data.results[mk].roc) return;
    const roc = data.results[mk].roc;
    datasets.push({
      label: `${LABELS[mk]} (AUC=${data.results[mk].roc_auc.toFixed(3)})`,
      data: roc.fpr.map((x,i) => ({x, y:roc.tpr[i]})),
      borderColor: COLORS[mk].border,
      backgroundColor: 'transparent',
      tension: .1, pointRadius: 0, borderWidth: 2,
    });
  });
  // diagonal
  datasets.push({
    label:'Chance', data:[{x:0,y:0},{x:1,y:1}],
    borderColor:'#334155', borderDash:[4,4], pointRadius:0, borderWidth:1,
  });
  rocChart = new Chart(document.getElementById('rocChart'), {
    type:'line',
    data:{ datasets },
    options:{
      responsive:true, maintainAspectRatio:true,
      plugins:{ legend:{ labels:{ color:'#94a3b8', font:{family:'Space Mono',size:9} } } },
      scales:{
        x:{ type:'linear', min:0, max:1, title:{display:true,text:'FPR',color:'#64748b'},
            ticks:{color:'#64748b'}, grid:{color:'rgba(255,255,255,.05)'} },
        y:{ type:'linear', min:0, max:1, title:{display:true,text:'TPR',color:'#64748b'},
            ticks:{color:'#64748b'}, grid:{color:'rgba(255,255,255,.05)'} }
      }
    }
  });
}

// ── FEATURES ──────────────────────────────────────────────────────────────────
function renderFeatures(data) {
  if (featChart) featChart.destroy();
  if (!data.feature_importance) return;
  const fi = data.feature_importance;
  const sorted = fi.names.map((n,i)=>({n,v:fi.values[i]})).sort((a,b)=>b.v-a.v).slice(0,10);
  featChart = new Chart(document.getElementById('featChart'), {
    type:'bar',
    data:{
      labels: sorted.map(x=>x.n),
      datasets:[{
        label:'Importancia',
        data: sorted.map(x=>x.v),
        backgroundColor: sorted.map((_,i)=>`hsla(${260+i*12},70%,65%,.7)`),
        borderRadius:4,
      }]
    },
    options:{
      indexAxis:'y', responsive:true, maintainAspectRatio:true,
      plugins:{ legend:{ display:false } },
      scales:{
        x:{ ticks:{color:'#64748b'}, grid:{color:'rgba(255,255,255,.05)'} },
        y:{ ticks:{color:'#94a3b8',font:{size:9}} }
      }
    }
  });
}

// ── CONFUSION MATRICES ────────────────────────────────────────────────────────
function renderCM(data) {
  const grid = document.getElementById('cmGrid');
  grid.innerHTML = '';
  const mkeys = Object.keys(data.results);
  mkeys.forEach(mk => {
    const cm = data.results[mk].confusion_matrix;
    const n  = cm.length;
    const max = Math.max(...cm.flat());
    let rows = '<tr><th></th>';
    for(let j=0;j<n;j++) rows += `<th>Pred ${j}</th>`;
    rows += '</tr>';
    for(let i=0;i<n;i++){
      rows += `<tr><th>Real ${i}</th>`;
      for(let j=0;j<n;j++){
        const v = cm[i][j];
        const ratio = v/max;
        const cls = ratio>.6?'cm-cell-high':ratio>.2?'cm-cell-med':'cm-cell-low';
        rows += `<td class="${cls}">${v}</td>`;
      }
      rows += '</tr>';
    }
    grid.innerHTML += `
      <div class="cm-card">
        <h4>${LABELS[mk]}</h4>
        <table class="cm-table">${rows}</table>
      </div>`;
  });
}

// ── CLASSIFICATION REPORT ─────────────────────────────────────────────────────
function renderReport(data) {
  const body = document.getElementById('reportBody');
  body.innerHTML = '';
  const mkeys = Object.keys(data.results);
  mkeys.forEach(mk => {
    const rep = data.results[mk].report;
    const classNames = ['class_0','class_1','class_2'];
    classNames.forEach((cls, idx) => {
      const r = rep[cls] || {};
      body.innerHTML += `<tr>
        <td><span class="badge badge-${mk}">${LABELS[mk]}</span></td>
        <td>Clase ${idx}</td>
        <td>${(r.precision||0).toFixed(3)}</td>
        <td>${(r.recall||0).toFixed(3)}</td>
        <td>${(r['f1-score']||0).toFixed(3)}</td>
        <td>${r.support||0}</td>
      </tr>`;
    });
    // macro avg
    const ma = rep['macro avg'] || {};
    body.innerHTML += `<tr style="background:rgba(255,255,255,.02)">
      <td><span class="badge badge-${mk}">${LABELS[mk]}</span></td>
      <td><em>macro avg</em></td>
      <td>${(ma.precision||0).toFixed(3)}</td>
      <td>${(ma.recall||0).toFixed(3)}</td>
      <td>${(ma['f1-score']||0).toFixed(3)}</td>
      <td>${ma.support||0}</td>
    </tr>`;
  });
}

// Run on load with defaults
window.onload = () => runExperiment();
</script>
</body>
</html>
"""

# ── ML Logic ──────────────────────────────────────────────────────────────────
def get_roc_data(clf, X_test, y_test, n_classes):
    """Compute macro-average ROC for multiclass."""
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    try:
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)
        else:
            y_score = clf.decision_function(X_test)
        # compute per-class then macro
        from sklearn.metrics import roc_curve, auc
        fpr_all, tpr_all = [], []
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_score[:, i])
            fpr_all.append(fpr_i)
            tpr_all.append(tpr_i)
        # interpolate on common grid
        all_fpr = np.unique(np.concatenate(fpr_all))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
        mean_tpr /= n_classes
        # downsample to 100 pts for JSON
        idx = np.linspace(0, len(all_fpr)-1, 100, dtype=int)
        return {"fpr": all_fpr[idx].tolist(), "tpr": mean_tpr[idx].tolist()}
    except Exception:
        return None


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/run", methods=["POST"])
def run():
    try:
        cfg = request.get_json()
        max_depth = int(cfg.get("max_depth", 3))
        cv_folds  = int(cfg.get("cv_folds", 5))
        n_est     = int(cfg.get("n_est", 50))
        selected  = cfg.get("models", {})

        # ── Dataset ─────────────────────────────────────────────────────────
        wine = load_wine()
        X, y = wine.data, wine.target
        feat_names = wine.feature_names
        n_classes  = len(np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.25, random_state=42, stratify=y)

        # ── Model catalogue ──────────────────────────────────────────────────
        base_dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        catalogue = {
            "dt":    DecisionTreeClassifier(max_depth=max_depth, random_state=42),
            "bag":   BaggingClassifier(
                         estimator=DecisionTreeClassifier(max_depth=max_depth),
                         n_estimators=n_est, random_state=42, n_jobs=-1),
            "boost": AdaBoostClassifier(
                         estimator=DecisionTreeClassifier(max_depth=min(max_depth, 3)),
                         n_estimators=n_est, random_state=42),
            "gb":    GradientBoostingClassifier(
                         max_depth=max_depth, n_estimators=n_est, random_state=42),
        }

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        results = {}
        for key, clf in catalogue.items():
            if not selected.get(key, False):
                continue

            # Cross-validation (same splits for all → fair comparison)
            cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

            # Train on train split → evaluate on test split
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

            # ROC-AUC (macro OvR)
            try:
                y_prob = clf.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            except Exception:
                roc_auc = 0.0

            roc_data = get_roc_data(clf, X_test, y_test, n_classes)

            cm = confusion_matrix(y_test, y_pred).tolist()
            rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            # rename keys to safe strings
            safe_rep = {}
            for k, v in rep.items():
                safe_rep[k.replace(" ", "_")] = v

            results[key] = {
                "accuracy":  acc,
                "precision": prec,
                "recall":    rec,
                "f1":        f1,
                "roc_auc":   roc_auc,
                "cv_scores": cv_scores.tolist(),
                "cv_mean":   float(cv_scores.mean()),
                "cv_std":    float(cv_scores.std()),
                "confusion_matrix": cm,
                "report": safe_rep,
                "roc": roc_data,
            }

        # Feature importance (Decision Tree if selected)
        fi = None
        if "dt" in results:
            dt_clf = catalogue["dt"]
            fi = {
                "names":  [f.replace("_", " ") for f in feat_names],
                "values": dt_clf.feature_importances_.tolist()
            }
        elif "gb" in results:
            gb_clf = catalogue["gb"]
            fi = {
                "names":  [f.replace("_", " ") for f in feat_names],
                "values": gb_clf.feature_importances_.tolist()
            }

        return jsonify({"results": results, "feature_importance": fi})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
