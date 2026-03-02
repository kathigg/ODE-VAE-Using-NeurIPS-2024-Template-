const RUNS_INDEX_CSV = "../neuroscience/runs/index.csv";

const runSelect = document.getElementById("runSelect");
const filterInput = document.getElementById("filterInput");
const reloadBtn = document.getElementById("reloadBtn");
const statusEl = document.getElementById("status");

const runDetails = document.getElementById("runDetails");
const metricSelect = document.getElementById("metricSelect");
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const speedSelect = document.getElementById("speedSelect");
const epochSlider = document.getElementById("epochSlider");
const epochLabel = document.getElementById("epochLabel");
const epochMetrics = document.getElementById("epochMetrics");
const curveCanvas = document.getElementById("curveCanvas");
const curveStatus = document.getElementById("curveStatus");
const lossCanvas = document.getElementById("lossCanvas");
const lossStatus = document.getElementById("lossStatus");

const modelDiagram = document.getElementById("modelDiagram");

const fileViewer = document.getElementById("fileViewer");
const tabRunJson = document.getElementById("tabRunJson");
const tabConfig = document.getElementById("tabConfig");
const tabConfigOrig = document.getElementById("tabConfigOrig");
const tabRunLog = document.getElementById("tabRunLog");
const tabGitDiff = document.getElementById("tabGitDiff");

const previewImg = document.getElementById("previewImg");
const manifoldImg = document.getElementById("manifoldImg");

const recordBtn = document.getElementById("recordBtn");
const stopRecordBtn = document.getElementById("stopRecordBtn");
const snapshotBtn = document.getElementById("snapshotBtn");
const recordStatus = document.getElementById("recordStatus");

let allRuns = [];
let filteredRuns = [];
let currentRunId = null;
let currentRunJson = null;
let currentConfig = null;
let currentConfigText = null;
let currentRunLogText = null;
let currentLogRecords = null;
let currentRunMeta = null;
let playbackTimer = null;
let playbackEpoch = null;
let fileCache = new Map();
let selectedLayerKey = null;
let currentArch = null;

let exportCanvas = null;
let exportCtx = null;
let isRecording = false;
let mediaRecorder = null;
let recordedChunks = [];

function parseCsv(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n").filter((l) => l.trim().length > 0);
  if (lines.length === 0) return [];
  const header = parseCsvLine(lines[0]).map((h) => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const fields = parseCsvLine(lines[i]);
    const row = {};
    for (let c = 0; c < header.length; c += 1) row[header[c]] = fields[c] ?? "";
    rows.push(row);
  }
  return rows;
}

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        cur += ch;
      }
    } else {
      if (ch === ",") {
        out.push(cur);
        cur = "";
      } else if (ch === '"') {
        inQuotes = true;
      } else {
        cur += ch;
      }
    }
  }
  out.push(cur);
  return out;
}

function parseConfig(text) {
  const cfg = {};
  for (const rawLine of text.replace(/\r\n/g, "\n").split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const idx = line.indexOf("=");
    if (idx < 0) continue;
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim();
    cfg[key] = value;
  }
  return cfg;
}

function safeNum(x) {
  const v = Number.parseFloat(String(x));
  return Number.isFinite(v) ? v : null;
}

function fmtNum(x, digits = 4) {
  if (x === null || x === undefined) return "—";
  const v = typeof x === "number" ? x : safeNum(x);
  if (v === null) return "—";
  return v.toFixed(digits);
}

function cfgNum(cfg, key, fallback = null) {
  if (!cfg || cfg[key] === undefined) return fallback;
  const v = safeNum(cfg[key]);
  return v === null ? fallback : v;
}

function cfgInt(cfg, key, fallback = null) {
  if (!cfg || cfg[key] === undefined) return fallback;
  const v = Number.parseInt(String(cfg[key]).trim(), 10);
  return Number.isFinite(v) ? v : fallback;
}

function escHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtCount(n) {
  if (!Number.isFinite(n)) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(2)}K`;
  return String(Math.round(n));
}

function setStatus(msg, kind = "info") {
  statusEl.textContent = msg;
  statusEl.style.color = kind === "error" ? "var(--danger)" : "var(--muted)";
}

function setCurveStatus(msg, kind = "info") {
  curveStatus.textContent = msg;
  curveStatus.style.color = kind === "error" ? "var(--danger)" : "var(--muted)";
}

function setLossStatus(msg, kind = "info") {
  lossStatus.textContent = msg;
  lossStatus.style.color = kind === "error" ? "var(--danger)" : "var(--muted)";
}

function applyFilter() {
  const q = filterInput.value.trim().toLowerCase();
  if (!q) {
    filteredRuns = allRuns.slice();
  } else {
    filteredRuns = allRuns.filter((r) => {
      const hay = `${r.run_id ?? ""} ${r.note ?? ""} ${r.commit ?? ""}`.toLowerCase();
      return hay.includes(q);
    });
  }
  renderRunList();
}

function renderRunList() {
  runSelect.innerHTML = "";
  for (const r of filteredRuns) {
    const opt = document.createElement("option");
    const rStr = fmtNum(r.r, 4);
    const r2Str = fmtNum(r.r2, 4);
    const note = (r.note ?? "").replace(/\s+/g, " ").trim();
    opt.value = r.run_id;
    opt.textContent = `${r.run_id}  r=${rStr}  R²=${r2Str}${note ? "  —  " + note : ""}`;
    runSelect.appendChild(opt);
  }
  setStatus(`Loaded ${filteredRuns.length} runs.`);
}

async function loadRunsIndex() {
  setStatus("Loading runs index…");
  const res = await fetch(RUNS_INDEX_CSV, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch ${RUNS_INDEX_CSV}: ${res.status}`);
  const csvText = await res.text();
  const rows = parseCsv(csvText);
  allRuns = rows.map((r) => ({
    ...r,
    r: safeNum(r.r),
    r2: safeNum(r.r2),
  }));
  filteredRuns = allRuns.slice();
  renderRunList();
}

function setImgOrMissing(imgEl, src) {
  imgEl.removeAttribute("data-missing");
  imgEl.onload = () => {
    imgEl.removeAttribute("data-missing");
    if (isRecording) drawExportFrame();
  };
  imgEl.onerror = () => {
    imgEl.src = "";
    imgEl.setAttribute("data-missing", "1");
    imgEl.alt = "(missing)";
    if (isRecording) drawExportFrame();
  };
  imgEl.src = src;
}

function renderDetails(runJson, cfg) {
  const kv = [];
  if (runJson) {
    kv.push(["run_id", runJson.run_id ?? currentRunId]);
    kv.push(["timestamp", runJson.timestamp ?? "—"]);
    kv.push(["note", runJson.note ?? "—"]);
    kv.push(["git_commit", runJson.git_commit ?? "—"]);
    if (runJson.metrics) {
      kv.push(["metrics.r", fmtNum(runJson.metrics.r, 4)]);
      kv.push(["metrics.R²", fmtNum(runJson.metrics.r2, 4)]);
    }
  } else {
    kv.push(["run_id", currentRunId]);
  }

  if (cfg) {
    const keys = [
      "latent_dim",
      "use_pca",
      "eval_metrics_pca",
      "no_leakage_preproc",
      "mind_split_enabled",
      "mind_test_frac",
      "holdout_trials",
      "landmark_count",
      "ode_solver",
      "epochs",
      "batch_size",
    ];
    for (const k of keys) {
      if (cfg[k] !== undefined) kv.push([`config.${k}`, cfg[k]]);
    }
  }

  runDetails.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "details__grid";
  for (const [k, v] of kv) {
    const kEl = document.createElement("div");
    kEl.className = "details__k";
    kEl.textContent = k;
    const vEl = document.createElement("div");
    vEl.className = "details__v";
    vEl.textContent = String(v);
    grid.appendChild(kEl);
    grid.appendChild(vEl);
  }
  runDetails.appendChild(grid);
}

function parseRunMeta(text) {
  if (!text) return null;
  const meta = {};
  for (const rawLine of text.replace(/\r\n/g, "\n").split("\n")) {
    const line = rawLine.trim();
    let m = null;

    m = /^PCA reduced\s+(\d+)\s*[→>-]\s*(\d+)\s+dims/i.exec(line);
    if (m) {
      meta.raw_dim = Number.parseInt(m[1], 10);
      meta.pca_dim = Number.parseInt(m[2], 10);
      continue;
    }

    m = /^Built sequences:\s*B=(\d+),\s*:?=?(\d+),\s*N=(\d+)/i.exec(line);
    if (m) {
      meta.batch_trials = Number.parseInt(m[1], 10);
      meta.seq_len = Number.parseInt(m[2], 10);
      meta.obs_dim = Number.parseInt(m[3], 10);
      continue;
    }

    m = /^MIND split:\s*train trials=(\d+)\s*\|\s*test trials=(\d+)/i.exec(line);
    if (m) {
      meta.mind_train_trials = Number.parseInt(m[1], 10);
      meta.mind_test_trials = Number.parseInt(m[2], 10);
      continue;
    }
  }

  return Object.keys(meta).length ? meta : null;
}

function metricToActiveTerm(metricPath) {
  if (!metricPath) return null;
  if (metricPath.startsWith("train.")) return metricPath.slice("train.".length);
  return null;
}

function pill(name, activeTerm, text, muted = false) {
  const active = name && activeTerm && name === activeTerm;
  const cls = ["pill"];
  if (active) cls.push("pill--active");
  if (muted) cls.push("pill--muted");
  return `<span class="${cls.join(" ")}">${text}</span>`;
}

function paramLinear(inDim, outDim, bias = true) {
  if (!Number.isFinite(inDim) || !Number.isFinite(outDim)) return null;
  return inDim * outDim + (bias ? outDim : 0);
}

function paramLayerNorm(dim) {
  if (!Number.isFinite(dim)) return null;
  return 2 * dim; // gamma + beta
}

function paramEmbedding(rows, dim) {
  if (!Number.isFinite(rows) || !Number.isFinite(dim)) return null;
  return rows * dim;
}

function unitsViz(units) {
  if (!Number.isFinite(units) || units <= 0) {
    return `<div class="unitsViz unitsViz--unknown"><span class="unitsViz__label">—</span></div>`;
  }
  const maxDots = units <= 12 ? units : units <= 64 ? 12 : 10;
  const dots = Array.from({ length: maxDots }, () => `<span class="unitDot"></span>`).join("");
  const label = units <= 12 ? `${units}` : `${units}`;
  return `<div class="unitsViz" aria-label="${units} units">${dots}<span class="unitsViz__label">${label}</span></div>`;
}

function layerCardHtml(layer) {
  const title = layer.title ?? "layer";
  const subtitleParts = [];
  if (layer.shape) subtitleParts.push(layer.shape);
  if (layer.activation) subtitleParts.push(layer.activation);
  if (layer.note) subtitleParts.push(layer.note);
  const subtitle = subtitleParts.filter(Boolean).join("  •  ");
  const params = Number.isFinite(layer.params) ? `${fmtCount(layer.params)} params` : null;
  const params2 = Number.isFinite(layer.params2) ? `${fmtCount(layer.params2)} params` : null;
  const lines = [subtitle, params, params2].filter(Boolean);

  const tooltip = layer.tooltip ?? "";
  const keyAttr = layer.key ? ` data-layer-key="${escHtml(layer.key)}"` : "";
  const activeClass = layer.key && selectedLayerKey && layer.key === selectedLayerKey ? " layerCard--active" : "";
  return `
    <div class="layerCard${activeClass}" role="button" tabindex="0"${keyAttr} title="${escHtml(tooltip)}">
      <div class="layerCard__top">
        <div class="layerCard__title">${escHtml(title)}</div>
        ${layer.units !== undefined && layer.units !== null ? unitsViz(layer.units) : ""}
      </div>
      ${lines.length ? `<div class="layerCard__sub">${lines.map(escHtml).join("<br/>")}</div>` : ""}
    </div>
  `;
}

function colHtml(col) {
  const layersHtml = (col.layers ?? []).map((l) => layerCardHtml(l)).join("");
  const subtitle = col.subtitle ? `<div class="archCol__sub">${escHtml(col.subtitle)}</div>` : "";
  return `
    <div class="archCol">
      <div class="archCol__title">${escHtml(col.title)}</div>
      ${subtitle}
      <div class="archCol__layers">${layersHtml}</div>
    </div>
  `;
}

function buildArchitecture(cfg, meta) {
  if (!cfg) return null;

  const latentDim = cfgInt(cfg, "latent_dim", null);
  const obsDim = meta?.obs_dim ?? meta?.pca_dim ?? meta?.raw_dim ?? null;
  const rawDim = meta?.raw_dim ?? null;
  const pcaDim = meta?.pca_dim ?? null;
  const seqLen = meta?.seq_len ?? null;

  const encTypeRaw = String(cfg?.encoder_type ?? "first").trim().toLowerCase();
  const decTypeRaw = String(cfg?.decoder_type ?? "mlp").trim().toLowerCase();

  const cols = [];

  const dataSub = [
    obsDim ? `N=${obsDim}` : null,
    seqLen ? `L=${seqLen}` : null,
    rawDim && pcaDim ? `raw ${rawDim} → PCA ${pcaDim}` : null,
  ]
    .filter(Boolean)
    .join("  •  ");

  cols.push({
    title: "Input",
    subtitle: dataSub || "x(t)",
    layers: [{ title: "x(t)", units: obsDim, note: "sequence", explain: "Observed ROI activity over time." }],
  });

  // --- Encoder ---
  const encoderLayers = [];
  if (encTypeRaw === "gru") {
    const hidden = cfgInt(cfg, "encoder_hidden", 256);
    const layers = cfgInt(cfg, "encoder_layers", 1);
    const bidir = String(cfg?.encoder_bidirectional ?? "false").trim().toLowerCase() === "true";
    const dirs = bidir ? 2 : 1;
    const in0 = obsDim;
    const h = hidden;

    // GRU param count (approx, PyTorch-style): per layer per direction:
    // 3H*in + 3H*H + 6H
    let gruParams = 0;
    for (let li = 0; li < (layers ?? 1); li += 1) {
      const inDim = li === 0 ? in0 : h * dirs;
      const perDir = 3 * h * inDim + 3 * h * h + 6 * h;
      gruParams += perDir * dirs;
    }
    const outDim = h * dirs;
    const muParams = paramLinear(outDim, latentDim, true);
    const lvParams = paramLinear(outDim, latentDim, true);

    encoderLayers.push({
      title: `GRU ×${layers ?? 1}${bidir ? " (bi)" : ""}`,
      units: hidden,
      shape: obsDim && hidden ? `${obsDim} → ${hidden}` : null,
      params: Number.isFinite(gruParams) ? gruParams : null,
      note: `hidden=${hidden}`,
      tooltip: "Sequence encoder over x(t).",
      explain: "Summarizes the full time-series into a single feature vector.",
    });
    encoderLayers.push({
      title: "μ head",
      units: latentDim,
      shape: latentDim ? `${outDim} → ${latentDim}` : null,
      params: muParams,
      explain: "Mean of the latent initial-state distribution.",
    });
    encoderLayers.push({
      title: "logσ² head",
      units: latentDim,
      shape: latentDim ? `${outDim} → ${latentDim}` : null,
      params: lvParams,
      explain: "Log-variance of the latent initial-state distribution.",
    });
  } else if (encTypeRaw === "transformer" || encTypeRaw === "trans") {
    const H = cfgInt(cfg, "encoder_hidden", 256);
    const L = cfgInt(cfg, "encoder_layers", 2);
    const heads = cfgInt(cfg, "encoder_heads", 4);
    const ffn = cfgInt(cfg, "encoder_ffn_dim", 512);
    const pool = String(cfg?.encoder_pool ?? "mean");

    encoderLayers.push({
      title: "Linear proj",
      units: H,
      shape: obsDim && H ? `${obsDim} → ${H}` : null,
      params: paramLinear(obsDim, H, true),
      tooltip: "Per-time-step projection to Transformer hidden size.",
      explain: "Maps each time step into the Transformer’s hidden dimension.",
    });
    encoderLayers.push({
      title: "Positional encoding",
      units: H,
      note: "sin/cos (no params)",
      explain: "Adds time information so attention can use temporal order.",
    });

    // Approx params per TransformerEncoderLayer (PyTorch):
    // MHA: 3H*H + 3H + H*H + H
    // FFN: H*F + F + F*H + H
    // LN: 4H (2 LNs)
    const perLayer =
      (Number.isFinite(H) && Number.isFinite(ffn)
        ? 3 * H * H + 3 * H + H * H + H + H * ffn + ffn + ffn * H + H + 4 * H
        : null);
    const blockParams = perLayer && Number.isFinite(L) ? perLayer * L : perLayer;

    encoderLayers.push({
      title: `Transformer blocks ×${L ?? 1}`,
      units: H,
      note: `heads=${heads ?? "—"}  ffn=${ffn ?? "—"}  pool=${pool}`,
      params: blockParams,
      tooltip: "Stack of TransformerEncoderLayer blocks (MHA + FFN + LayerNorm).",
      explain: "Self-attention + feedforward layers that model temporal dependencies.",
    });

    encoderLayers.push({
      title: "μ head",
      units: latentDim,
      shape: latentDim ? `${H} → ${latentDim}` : null,
      params: paramLinear(H, latentDim, true),
      explain: "Mean of the latent initial-state distribution.",
    });
    encoderLayers.push({
      title: "logσ² head",
      units: latentDim,
      shape: latentDim ? `${H} → ${latentDim}` : null,
      params: paramLinear(H, latentDim, true),
      explain: "Log-variance of the latent initial-state distribution.",
    });
  } else if (encTypeRaw === "attention" || encTypeRaw === "attn" || encTypeRaw === "temporal_attn") {
    const hidden = cfgInt(cfg, "encoder_hidden", 256);
    const projParams = paramLinear(obsDim, hidden, true);
    const scoreParams =
      (paramLinear(hidden, hidden, true) ?? 0) + (paramLinear(hidden, 1, true) ?? 0);
    encoderLayers.push({
      title: "Temporal attention",
      units: hidden,
      note: `hidden=${hidden}`,
      params: ((projParams ?? 0) + (scoreParams ?? 0)) || null,
      tooltip: "Attention pooling over time with learned scores.",
      explain: "Learns which time points matter most, then pools them into one vector.",
    });
    encoderLayers.push({
      title: "μ head",
      units: latentDim,
      shape: latentDim ? `${hidden} → ${latentDim}` : null,
      params: paramLinear(hidden, latentDim, true),
      explain: "Mean of the latent initial-state distribution.",
    });
    encoderLayers.push({
      title: "logσ² head",
      units: latentDim,
      shape: latentDim ? `${hidden} → ${latentDim}` : null,
      params: paramLinear(hidden, latentDim, true),
      explain: "Log-variance of the latent initial-state distribution.",
    });
  } else {
    // default first-frame MLP encoder
    const p1 = paramLinear(obsDim, 512, true);
    const p2 = paramLinear(512, 256, true);
    const p3 = paramLinear(256, 128, true);
    const muP = paramLinear(128, latentDim, true);
    const lvP = paramLinear(128, latentDim, true);

    encoderLayers.push({
      title: "Linear",
      units: 512,
      shape: obsDim ? `${obsDim} → 512` : null,
      activation: "ReLU",
      params: p1,
      explain: "First encoder layer (per trial/frame).",
    });
    encoderLayers.push({ title: "Linear", units: 256, shape: "512 → 256", activation: "ReLU", params: p2 });
    encoderLayers.push({ title: "Linear", units: 128, shape: "256 → 128", activation: "ReLU", params: p3 });
    encoderLayers.push({
      title: "μ head",
      units: latentDim,
      shape: latentDim ? `128 → ${latentDim}` : null,
      params: muP,
      explain: "Mean of the latent initial-state distribution.",
    });
    encoderLayers.push({
      title: "logσ² head",
      units: latentDim,
      shape: latentDim ? `128 → ${latentDim}` : null,
      params: lvP,
      explain: "Log-variance of the latent initial-state distribution.",
    });
  }

  encoderLayers.push({
    title: "Reparameterize",
    units: latentDim,
    note: "z₀ = μ + σ⊙ε",
    explain: "Sampling trick that keeps training differentiable.",
  });

  cols.push({
    title: "Encoder",
    subtitle: `type=${encTypeRaw}`,
    layers: encoderLayers,
  });

  cols.push({
    title: "Latent init",
    subtitle: latentDim ? `D=${latentDim}` : "z₀",
    layers: [{ title: "z₀", units: latentDim, note: "sampled", explain: "Starting point for latent dynamics." }],
  });

  // --- ODE function (MoE) ---
  const numExperts = cfgInt(cfg, "num_experts", 4);
  const odeHidden = cfgInt(cfg, "hidden", 128);
  const odeTimeDep = String(cfg?.ode_time_dependent ?? "false").trim().toLowerCase() === "true";
  const odeTimeEmb = cfgInt(cfg, "ode_time_embed_dim", 16);
  const odeSolver = String(cfg?.ode_solver ?? "dopri5");
  const odeIn = latentDim + (odeTimeDep ? odeTimeEmb : 0);

  const gateParams = (paramLinear(odeIn, odeHidden, true) ?? 0) + (paramLinear(odeHidden, numExperts, true) ?? 0);
  const expertParamsPer =
    (paramLinear(odeIn, odeHidden, true) ?? 0) +
    (paramLinear(odeHidden, odeHidden, true) ?? 0) +
    (paramLinear(odeHidden, latentDim, true) ?? 0);
  const expertsParams = Number.isFinite(expertParamsPer) && Number.isFinite(numExperts) ? expertParamsPer * numExperts : null;
  const timeMlpParams = odeTimeDep
    ? (paramLinear(1, odeTimeEmb, true) ?? 0) + (paramLinear(odeTimeEmb, odeTimeEmb, true) ?? 0)
    : null;

  const odeLayers = [];
  if (odeTimeDep) {
    odeLayers.push({
      title: "Time embed",
      units: odeTimeEmb,
      note: "MLP(t)",
      params: timeMlpParams,
      explain: "Optional: lets dynamics change explicitly with time.",
    });
  }
  odeLayers.push({
    title: "Gate",
    units: numExperts,
    note: "softmax over experts",
    params: gateParams,
    tooltip: "Gating network producing mixture weights π_e(z).",
    explain: "Chooses (softly) which expert dynamics to use.",
  });
  odeLayers.push({
    title: `Experts ×${numExperts}`,
    units: latentDim,
    note: `MLP hidden=${odeHidden}`,
    params: expertsParams,
    tooltip: "Each expert outputs dz/dt; combined by gate weights.",
    explain: "Each expert proposes a different latent “direction of change”.",
  });
  odeLayers.push({
    title: "LayerNorm",
    units: latentDim,
    params: paramLayerNorm(latentDim),
  });
  odeLayers.push({
    title: "Integrate",
    units: seqLen,
    note: `solver=${odeSolver}`,
    tooltip: "ODE solver integrates z(t) over the time grid.",
    explain: "Produces the full latent trajectory z(t).",
  });

  cols.push({
    title: "Latent ODE",
    subtitle: "MoE vector field",
    layers: odeLayers,
  });

  // --- Decoder ---
  const decoderLayers = [];
  if (decTypeRaw === "moe") {
    const E = cfgInt(cfg, "dec_num_experts", 4);
    const hidden = cfgInt(cfg, "decoder_hidden", 256);
    const perExpert =
      (paramLinear(latentDim, hidden, true) ?? 0) +
      (paramLinear(hidden, hidden, true) ?? 0) +
      (paramLinear(hidden, obsDim, true) ?? 0);
    decoderLayers.push({
      title: `Experts ×${E}`,
      units: obsDim,
      note: `MLP hidden=${hidden}`,
      params: Number.isFinite(perExpert) && Number.isFinite(E) ? perExpert * E : null,
      tooltip: "Each expert decodes z(t) to x̂(t).",
      explain: "Multiple decoders; the model combines them for each neuron.",
    });
    decoderLayers.push({
      title: "Neuron mixing logits",
      units: E,
      note: "softmax per neuron",
      params: paramEmbedding(obsDim, E),
      tooltip: "Per-neuron mixture weights over decoder experts (N×E).",
      explain: "Each neuron has its own mixture over decoder experts.",
    });
  } else if (decTypeRaw === "neuronaware") {
    const embDim = cfgInt(cfg, "decoder_emb_dim", 16);
    const hidden = cfgInt(cfg, "decoder_hidden", 256);
    decoderLayers.push({
      title: "Neuron embedding",
      units: embDim,
      note: obsDim ? `table N×${embDim}` : null,
      params: paramEmbedding(obsDim, embDim),
      explain: "Gives each neuron an identity vector the decoder can use.",
    });
    const inD = (latentDim ?? 0) + (embDim ?? 0);
    const mlpParams =
      (paramLinear(inD, hidden, true) ?? 0) +
      (paramLinear(hidden, hidden, true) ?? 0) +
      (paramLinear(hidden, 1, true) ?? 0);
    decoderLayers.push({
      title: "Per-neuron MLP",
      units: hidden,
      note: `hidden=${hidden}`,
      params: mlpParams || null,
      tooltip: "Shared MLP applied per neuron (concatenate z(t) with neuron embedding).",
      explain: "Same MLP for all neurons; embeddings personalize outputs.",
    });
  } else if (decTypeRaw === "localattn") {
    const k = cfgInt(cfg, "k_neighbors", 16);
    const hidden = cfgInt(cfg, "decoder_hidden", 256);
    decoderLayers.push({
      title: "Neuron embedding",
      units: latentDim,
      note: obsDim && latentDim ? `table N×D` : null,
      params: paramEmbedding(obsDim, latentDim),
      tooltip: "Embeddings used to define neuron-neighborhood similarity.",
      explain: "Defines which neurons are ‘neighbors’ for local decoding.",
    });
    decoderLayers.push({ title: `Top‑k neighbors`, units: k, note: "per neuron" });
    const mlpParams =
      (paramLinear((latentDim ?? 0) * 2, hidden, true) ?? 0) +
      (paramLinear(hidden, hidden, true) ?? 0) +
      (paramLinear(hidden, 1, true) ?? 0);
    decoderLayers.push({
      title: "Per-neuron MLP",
      units: hidden,
      note: `hidden=${hidden}`,
      params: mlpParams || null,
      tooltip: "Shared MLP over concat(global z, local neighbor aggregate).",
      explain: "Decodes using both global state and local neighbor summary.",
    });
  } else {
    // default MLP decoder (as implemented in v5)
    decoderLayers.push({
      title: "Linear",
      units: 512,
      shape: latentDim ? `${latentDim} → 512` : null,
      activation: "ReLU",
      params: paramLinear(latentDim, 512, true),
      explain: "Maps latent state to a higher-dimensional feature space.",
    });
    decoderLayers.push({
      title: "Linear",
      units: 512,
      shape: "512 → 512",
      activation: "ReLU",
      params: paramLinear(512, 512, true),
    });
    decoderLayers.push({
      title: "Linear",
      units: obsDim,
      shape: obsDim ? `512 → ${obsDim}` : null,
      params: paramLinear(512, obsDim, true),
      explain: "Final projection to the observed ROI dimension.",
    });
  }

  cols.push({
    title: "Decoder",
    subtitle: `type=${decTypeRaw}`,
    layers: decoderLayers,
  });

  cols.push({
    title: "Output",
    subtitle: obsDim ? `N=${obsDim}` : "x̂(t)",
    layers: [{ title: "x̂(t)", units: obsDim, note: "reconstruction", explain: "Model’s predicted ROI activity." }],
  });

  const byKey = new Map();
  for (let ci = 0; ci < cols.length; ci += 1) {
    const col = cols[ci];
    for (let li = 0; li < (col.layers ?? []).length; li += 1) {
      const layer = col.layers[li];
      const key = `${col.title}:${li}:${layer.title}`;
      layer.key = key;
      byKey.set(key, { colTitle: col.title, ...layer });
    }
  }

  // compute totals
  let total = 0;
  for (const c of cols) {
    for (const l of c.layers ?? []) {
      if (Number.isFinite(l.params)) total += l.params;
      if (Number.isFinite(l.params2)) total += l.params2;
    }
  }
  return { cols, totalParams: total, byKey };
}

function renderModelDiagram(cfg, meta, activeTerm, cursorRec) {
  if (!modelDiagram) return;
  if (!cfg) {
    modelDiagram.textContent = "(config.txt missing)";
    return;
  }

  const arch = buildArchitecture(cfg, meta);
  currentArch = arch;
  const archHtml = arch
    ? `
      <div class="archSummary">
        <div class="archSummary__k">Approx params</div>
        <div class="archSummary__v">${fmtCount(arch.totalParams)} (from config + run.log dims)</div>
      </div>
      <div class="archFlow">
        ${arch.cols
          .map((c, i) => {
            const arrow = i === 0 ? "" : `<div class="archArrow">→</div>`;
            return `${arrow}${colHtml(c)}`;
          })
          .join("")}
      </div>
    `
    : `<div class="status status--muted">(could not build architecture)</div>`;

  const selected = selectedLayerKey && arch?.byKey ? arch.byKey.get(selectedLayerKey) : null;
  const layerDetailsHtml = selected
    ? `
      <div class="layerDetails__title">${escHtml(selected.colTitle)} / ${escHtml(selected.title ?? "layer")}</div>
      <div class="layerDetails__body">
        ${selected.explain ? `<div class="layerDetails__p">${escHtml(selected.explain)}</div>` : ""}
        ${selected.shape ? `<div class="layerDetails__p"><span class="layerDetails__k">shape</span> ${escHtml(selected.shape)}</div>` : ""}
        ${
          Number.isFinite(selected.params)
            ? `<div class="layerDetails__p"><span class="layerDetails__k">params</span> ${escHtml(fmtCount(selected.params))}</div>`
            : ""
        }
        ${selected.note ? `<div class="layerDetails__p"><span class="layerDetails__k">note</span> ${escHtml(selected.note)}</div>` : ""}
      </div>
    `
    : `<div class="layerDetails__empty">Click a layer above to see a plain-language explanation.</div>`;

  const trialLen = cfgNum(cfg, "trial_len_s", null);
  const fps = cfgNum(cfg, "fps", null);
  const holdoutTrials = cfgInt(cfg, "holdout_trials", null);

  const beta = cfgNum(cfg, "beta", null);
  const klWarmup = cfgInt(cfg, "kl_warmup_epochs", null);
  const lambdaSmooth = cfgNum(cfg, "lambda_smooth", null);
  const lambdaTrans = cfgNum(cfg, "lambda_transition", null);
  const lambdaTransWarmup = cfgInt(cfg, "lambda_transition_warmup_epochs", null);
  const lambdaLle = cfgNum(cfg, "lambda_lle", null);
  const lleK = cfgInt(cfg, "lle_k", null);

  const cursorLine = cursorRec
    ? `Epoch #${cursorRec.epoch}  beta=${fmtNum(cursorRec.train.beta, 3)}  valid.R²=${fmtNum(cursorRec.valid.r2, 4)}`
    : null;

  const objLine = `≈ recon + β·KL + λsmooth·smooth + λtrans(t)·trans + λlle·lle`;
  const objWeights = `β=${beta ?? "—"} (warmup=${klWarmup ?? "—"}),  λsmooth=${lambdaSmooth ?? "—"},  λtrans=${lambdaTrans ?? "—"} (warmup=${
    lambdaTransWarmup ?? "—"
  }),  λlle=${lambdaLle ?? "—"} (k=${lleK ?? "—"})`;

  const datasetLineParts = [];
  if (Number.isFinite(trialLen) && Number.isFinite(fps)) datasetLineParts.push(`trial=${trialLen}s @ ${fps}fps`);
  if (holdoutTrials !== null && holdoutTrials !== undefined) datasetLineParts.push(`holdout_trials=${holdoutTrials}`);
  if (meta?.raw_dim && meta?.pca_dim) datasetLineParts.push(`PCA: ${meta.raw_dim} → ${meta.pca_dim}`);
  const datasetLine = datasetLineParts.join("  •  ");

  modelDiagram.innerHTML = `
    <div class="diagram">
      ${datasetLine ? `<div class="status status--muted">${datasetLine}</div>` : ""}
      ${archHtml}
      <div class="layerDetails">${layerDetailsHtml}</div>
      <div class="node">
        <div class="node__title">Objective (train)</div>
        <div class="node__sub">${objLine}</div>
        <div class="node__sub">${objWeights}</div>
        ${cursorLine ? `<div class="node__sub">${cursorLine}</div>` : ""}
        <div class="lossPills">
          ${pill("loss", activeTerm, "train.loss")}
          ${pill("recon", activeTerm, "recon")}
          ${pill("kl", activeTerm, "β·KL")}
          ${pill("smooth", activeTerm, "λsmooth·smooth")}
          ${pill("trans", activeTerm, "λtrans(t)·trans")}
          ${pill("lle", activeTerm, "λlle·LLE")}
          ${pill("beta", activeTerm, "beta")}
        </div>
      </div>
    </div>
  `;
}

function parseRunLog(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const trainRe =
    /^\[(\d+)\]\s+train loss\s+([-\deE.]+)\s+\|\s+recon\s+([-\deE.]+)\s+\|\s+kl\s+([-\deE.]+)\s+\|\s+smooth\s+([-\deE.]+)\s+\|\s+trans\s+([-\deE.]+)\s+\|\s+lle\s+([-\deE.]+)\s+\|\s+beta\s+([-\deE.]+)/;
  const validRe =
    /^\s*valid loss\s+([-\deE.]+)\s+\|\s+recon\s+([-\deE.]+)\s+\|\s+kl\s+([-\deE.]+)\s+\|\s+smooth\s+([-\deE.]+)\s+\|\s+trans\s+([-\deE.]+)\s+\|\s+lle\s+([-\deE.]+)\s+\|\s+r\s+([-\deE.]+)\s+\|\s+R(?:²|2|\^2)\s+([-\deE.]+)/;

  const recs = [];
  let cur = null;
  for (const line of lines) {
    const mt = trainRe.exec(line);
    if (mt) {
      cur = {
        epoch: Number.parseInt(mt[1], 10),
        train: {
          loss: safeNum(mt[2]),
          recon: safeNum(mt[3]),
          kl: safeNum(mt[4]),
          smooth: safeNum(mt[5]),
          trans: safeNum(mt[6]),
          lle: safeNum(mt[7]),
          beta: safeNum(mt[8]),
        },
        valid: null,
      };
      recs.push(cur);
      continue;
    }
    const mv = validRe.exec(line);
    if (mv && cur) {
      cur.valid = {
        loss: safeNum(mv[1]),
        recon: safeNum(mv[2]),
        kl: safeNum(mv[3]),
        smooth: safeNum(mv[4]),
        trans: safeNum(mv[5]),
        lle: safeNum(mv[6]),
        r: safeNum(mv[7]),
        r2: safeNum(mv[8]),
      };
      cur = null;
    }
  }
  return recs.filter((r) => r.valid);
}

function getMetric(records, path) {
  const [scope, key] = path.split(".");
  return records
    .map((r) => ({ x: r.epoch, y: r?.[scope]?.[key] ?? null }))
    .filter((p) => p.y !== null && Number.isFinite(p.y));
}

function getNearestRecordByEpoch(records, epoch) {
  if (!records || records.length === 0 || epoch === null || epoch === undefined) return null;
  let best = records[0];
  let bestDist = Math.abs(best.epoch - epoch);
  for (let i = 1; i < records.length; i += 1) {
    const d = Math.abs(records[i].epoch - epoch);
    if (d < bestDist) {
      bestDist = d;
      best = records[i];
    }
  }
  return best;
}

function renderEpochMetricsForEpoch(epoch) {
  epochMetrics.innerHTML = "";
  if (!currentLogRecords || currentLogRecords.length === 0 || epoch === null || epoch === undefined) return;
  const rec = getNearestRecordByEpoch(currentLogRecords, epoch);
  if (!rec) return;

  const cards = [
    ["epoch", `#${rec.epoch}`],
    ["train.loss", fmtNum(rec.train.loss, 5)],
    ["valid.loss", fmtNum(rec.valid.loss, 5)],
    ["valid.r", fmtNum(rec.valid.r, 4)],
    ["valid.R²", fmtNum(rec.valid.r2, 4)],
    ["train.recon", fmtNum(rec.train.recon, 5)],
    ["train.kl", fmtNum(rec.train.kl, 5)],
    ["train.beta", fmtNum(rec.train.beta, 3)],
  ];

  for (const [k, v] of cards) {
    const card = document.createElement("div");
    card.className = "metricCard";
    const kEl = document.createElement("div");
    kEl.className = "metricCard__k";
    kEl.textContent = k;
    const vEl = document.createElement("div");
    vEl.className = "metricCard__v";
    vEl.textContent = String(v);
    card.appendChild(kEl);
    card.appendChild(vEl);
    epochMetrics.appendChild(card);
  }
}

function ensureExportCanvas() {
  if (exportCanvas && exportCtx) return;
  exportCanvas = document.createElement("canvas");
  exportCanvas.width = 1280;
  exportCanvas.height = 720;
  exportCtx = exportCanvas.getContext("2d");
}

function drawLabeledBox(ctx, x, y, w, h, label) {
  ctx.fillStyle = "rgba(10, 14, 20, 0.85)";
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = "rgba(255,255,255,0.14)";
  ctx.lineWidth = 2;
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  ctx.fillStyle = "rgba(232,238,247,0.75)";
  ctx.font = "14px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  ctx.fillText(label, x + 12, y + 22);
}

function drawExportFrame() {
  if (!exportCtx) return;
  const ctx = exportCtx;
  const W = exportCanvas.width;
  const H = exportCanvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b0f14";
  ctx.fillRect(0, 0, W, H);

  const pad = 20;
  const headerH = 52;
  const gap = 16;
  const leftX = pad;
  const topY = headerH;
  const areaW = W - pad * 2;
  const areaH = H - headerH - pad;
  const cellW = (areaW - gap) / 2;
  const cellH = (areaH - gap) / 2;

  const epoch = (playbackEpoch ?? Number.parseInt(epochSlider.value, 10)) || null;
  const rec = getNearestRecordByEpoch(currentLogRecords, epoch);

  ctx.fillStyle = "rgba(232,238,247,0.92)";
  ctx.font = "18px ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial";
  const title = currentRunId ? `Run ${currentRunId}` : "Run";
  ctx.fillText(title, pad, 28);

  ctx.fillStyle = "rgba(232,238,247,0.72)";
  ctx.font = "13px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  const epochStr = rec ? `epoch #${rec.epoch}` : "epoch —";
  const metricStr = metricSelect?.value ? `metric=${metricSelect.value}` : "";
  const r2Str = rec?.valid?.r2 !== null && rec?.valid?.r2 !== undefined ? `valid.R²=${fmtNum(rec.valid.r2, 4)}` : "";
  const line = [epochStr, metricStr, r2Str].filter((s) => s && s.length > 0).join("  |  ");
  ctx.fillText(line, pad, 48);

  const x0 = leftX;
  const x1 = leftX + cellW + gap;
  const y0 = topY;
  const y1 = topY + cellH + gap;

  drawLabeledBox(ctx, x0, y0, cellW, cellH, "Learning curve");
  if (curveCanvas) ctx.drawImage(curveCanvas, x0 + 10, y0 + 32, cellW - 20, cellH - 44);

  drawLabeledBox(ctx, x0, y1, cellW, cellH, "Loss composition");
  if (lossCanvas) ctx.drawImage(lossCanvas, x0 + 10, y1 + 32, cellW - 20, cellH - 44);

  drawLabeledBox(ctx, x1, y0, cellW, cellH, "preview.png");
  if (previewImg && previewImg.naturalWidth > 0 && !previewImg.getAttribute("data-missing")) {
    ctx.drawImage(previewImg, x1 + 10, y0 + 32, cellW - 20, cellH - 44);
  } else {
    ctx.fillStyle = "rgba(232,238,247,0.65)";
    ctx.fillText("(missing)", x1 + 12, y0 + 56);
  }

  drawLabeledBox(ctx, x1, y1, cellW, cellH, "latent_manifold_mds.png");
  if (manifoldImg && manifoldImg.naturalWidth > 0 && !manifoldImg.getAttribute("data-missing")) {
    ctx.drawImage(manifoldImg, x1 + 10, y1 + 32, cellW - 20, cellH - 44);
  } else {
    ctx.fillStyle = "rgba(232,238,247,0.65)";
    ctx.fillText("(missing)", x1 + 12, y1 + 56);
  }
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawLine(canvas, pts, opts) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  ctx.scale(dpr, dpr);

  ctx.fillStyle = "rgba(10, 14, 20, 0.55)";
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 48;
  const padR = 16;
  const padT = 18;
  const padB = 34;
  const plotW = cssW - padL - padR;
  const plotH = cssH - padT - padB;

  if (!pts || pts.length < 2) {
    ctx.fillStyle = "rgba(255,255,255,0.78)";
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
    ctx.fillText("No curve data parsed from run.log.", padL, padT + 18);
    return;
  }

  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad;
  yMax += yPad;

  const xToPx = (x) => padL + ((x - xMin) / (xMax - xMin)) * plotW;
  const yToPx = (y) => padT + (1 - (y - yMin) / (yMax - yMin)) * plotH;

  // grid + axes
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const yy = padT + (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, yy);
    ctx.lineTo(padL + plotW, yy);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  // labels
  ctx.fillStyle = "rgba(255,255,255,0.78)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  ctx.fillText(`${opts?.title ?? "metric"} vs epoch`, padL, 14);

  ctx.fillStyle = "rgba(255,255,255,0.60)";
  ctx.fillText(String(xMin), padL, padT + plotH + 22);
  ctx.fillText(String(xMax), padL + plotW - 24, padT + plotH + 22);

  ctx.fillText(yMax.toFixed(3), 6, padT + 10);
  ctx.fillText(yMin.toFixed(3), 6, padT + plotH);

  // curve
  ctx.strokeStyle = "rgba(124,199,255,0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(xToPx(pts[0].x), yToPx(pts[0].y));
  for (let i = 1; i < pts.length; i += 1) ctx.lineTo(xToPx(pts[i].x), yToPx(pts[i].y));
  ctx.stroke();

  // cursor
  if (opts?.cursorEpoch !== null && opts?.cursorEpoch !== undefined) {
    const cx = Math.min(Math.max(opts.cursorEpoch, xMin), xMax);
    const cxPx = xToPx(cx);
    let nearest = pts[0];
    let bestDist = Math.abs(nearest.x - cx);
    for (let i = 1; i < pts.length; i += 1) {
      const d = Math.abs(pts[i].x - cx);
      if (d < bestDist) {
        bestDist = d;
        nearest = pts[i];
      }
    }

    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cxPx, padT);
    ctx.lineTo(cxPx, padT + plotH);
    ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.beginPath();
    ctx.arc(xToPx(nearest.x), yToPx(nearest.y), 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // endpoints
  ctx.fillStyle = "rgba(124,199,255,0.95)";
  for (const p of [pts[0], pts[pts.length - 1]]) {
    ctx.beginPath();
    ctx.arc(xToPx(p.x), yToPx(p.y), 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawMultiLine(canvas, seriesList, opts) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  ctx.scale(dpr, dpr);

  ctx.fillStyle = "rgba(10, 14, 20, 0.55)";
  ctx.fillRect(0, 0, cssW, cssH);

  const padL = 48;
  const padR = 16;
  const padT = 18;
  const padB = 34;
  const plotW = cssW - padL - padR;
  const plotH = cssH - padT - padB;

  const allPts = [];
  for (const s of seriesList ?? []) {
    for (const p of s.pts ?? []) allPts.push(p);
  }

  if (!allPts || allPts.length < 2) {
    ctx.fillStyle = "rgba(255,255,255,0.78)";
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
    ctx.fillText("No data.", padL, padT + 18);
    return;
  }

  const xs = allPts.map((p) => p.x);
  const ys = allPts.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad;
  yMax += yPad;

  const xToPx = (x) => padL + ((x - xMin) / (xMax - xMin)) * plotW;
  const yToPx = (y) => padT + (1 - (y - yMin) / (yMax - yMin)) * plotH;

  // grid + axes
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const yy = padT + (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, yy);
    ctx.lineTo(padL + plotW, yy);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  // labels
  ctx.fillStyle = "rgba(255,255,255,0.78)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  ctx.fillText(`${opts?.title ?? "metric"} vs epoch`, padL, 14);

  ctx.fillStyle = "rgba(255,255,255,0.60)";
  ctx.fillText(String(xMin), padL, padT + plotH + 22);
  ctx.fillText(String(xMax), padL + plotW - 24, padT + plotH + 22);
  ctx.fillText(yMax.toFixed(3), 6, padT + 10);
  ctx.fillText(yMin.toFixed(3), 6, padT + plotH);

  // cursor
  if (opts?.cursorEpoch !== null && opts?.cursorEpoch !== undefined) {
    const cx = Math.min(Math.max(opts.cursorEpoch, xMin), xMax);
    const cxPx = xToPx(cx);
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cxPx, padT);
    ctx.lineTo(cxPx, padT + plotH);
    ctx.stroke();
  }

  // series
  for (const s of seriesList ?? []) {
    const pts = s.pts ?? [];
    if (pts.length < 2) continue;
    ctx.strokeStyle = s.color ?? "rgba(124,199,255,0.95)";
    ctx.lineWidth = s.width ?? 2;
    ctx.beginPath();
    ctx.moveTo(xToPx(pts[0].x), yToPx(pts[0].y));
    for (let i = 1; i < pts.length; i += 1) ctx.lineTo(xToPx(pts[i].x), yToPx(pts[i].y));
    ctx.stroke();
  }

  // legend
  const legendX = padL + plotW - 8;
  let legendY = padT + 10;
  ctx.textAlign = "right";
  for (const s of seriesList ?? []) {
    ctx.fillStyle = s.color ?? "rgba(255,255,255,0.78)";
    ctx.fillText(s.name ?? "series", legendX, legendY);
    legendY += 14;
  }
  ctx.textAlign = "left";
}

function computeLossSeries(records, cfg) {
  const lambdaSmooth = cfgNum(cfg, "lambda_smooth", 0.0) ?? 0.0;
  const lambdaLle = cfgNum(cfg, "lambda_lle", 0.0) ?? 0.0;
  const lambdaTransFinal = cfgNum(cfg, "lambda_transition", 0.0) ?? 0.0;
  const transWarmup = cfgInt(cfg, "lambda_transition_warmup_epochs", 0) ?? 0;

  const recon = [];
  const betaKl = [];
  const smooth = [];
  const trans = [];
  const lle = [];
  const approx = [];
  const reported = [];

  for (const r of records ?? []) {
    const epoch = r.epoch;
    const beta = r.train.beta ?? 0.0;
    const lamTrans =
      transWarmup > 0 ? lambdaTransFinal * Math.min(1.0, epoch / Math.max(1, transWarmup)) : lambdaTransFinal;
    const reconC = r.train.recon ?? 0.0;
    const betaKlC = (r.train.kl ?? 0.0) * beta;
    const smoothC = (r.train.smooth ?? 0.0) * lambdaSmooth;
    const transC = (r.train.trans ?? 0.0) * lamTrans;
    const lleC = (r.train.lle ?? 0.0) * lambdaLle;
    const approxC = reconC + betaKlC + smoothC + transC + lleC;

    recon.push({ x: epoch, y: reconC });
    betaKl.push({ x: epoch, y: betaKlC });
    smooth.push({ x: epoch, y: smoothC });
    trans.push({ x: epoch, y: transC });
    lle.push({ x: epoch, y: lleC });
    approx.push({ x: epoch, y: approxC });
    reported.push({ x: epoch, y: r.train.loss ?? approxC });
  }

  return [
    { name: "train.loss (reported)", pts: reported, color: "rgba(255,255,255,0.85)", width: 2 },
    { name: "sum (approx)", pts: approx, color: "rgba(124,199,255,0.95)", width: 2 },
    { name: "recon", pts: recon, color: "rgba(124,199,255,0.65)", width: 1.5 },
    { name: "beta·KL", pts: betaKl, color: "rgba(255,209,102,0.80)", width: 1.5 },
    { name: "λsmooth·smooth", pts: smooth, color: "rgba(6,214,160,0.80)", width: 1.5 },
    { name: "λtrans(t)·trans", pts: trans, color: "rgba(239,71,111,0.80)", width: 1.5 },
    { name: "λlle·lle", pts: lle, color: "rgba(179,136,255,0.80)", width: 1.5 },
  ];
}

function setSelectedTab(btn) {
  for (const b of [tabRunJson, tabConfig, tabConfigOrig, tabRunLog, tabGitDiff]) {
    if (!b) continue;
    b.setAttribute("aria-selected", b === btn ? "true" : "false");
  }
}

async function showFile(name) {
  if (!currentRunId) {
    fileViewer.textContent = "(select a run to view files)";
    return;
  }

  if (fileCache.has(name)) {
    fileViewer.textContent = fileCache.get(name);
    return;
  }

  // Preloaded files
  if (name === "run.json" && currentRunJson) {
    const txt = JSON.stringify(currentRunJson, null, 2);
    fileCache.set(name, txt);
    fileViewer.textContent = txt;
    return;
  }
  if (name === "config.txt" && currentConfigText) {
    fileCache.set(name, currentConfigText);
    fileViewer.textContent = currentConfigText;
    return;
  }
  if (name === "run.log" && currentRunLogText !== null && currentRunLogText !== undefined) {
    fileCache.set(name, currentRunLogText);
    fileViewer.textContent = currentRunLogText || "(empty)";
    return;
  }

  const url = `../neuroscience/runs/${currentRunId}/${name}`;
  fileViewer.textContent = `Loading ${name}…`;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      fileViewer.textContent = `(missing) ${name} (HTTP ${res.status})`;
      return;
    }
    const txt = await res.text();
    fileCache.set(name, txt);
    fileViewer.textContent = txt || "(empty)";
  } catch (e) {
    fileViewer.textContent = `Failed to load ${name}: ${String(e?.message ?? e)}`;
  }
}

function stopPlayback() {
  if (playbackTimer) {
    window.clearInterval(playbackTimer);
    playbackTimer = null;
  }
  playBtn.disabled = false;
  pauseBtn.disabled = true;
}

function setEpoch(epoch) {
  if (!currentLogRecords || currentLogRecords.length === 0) return;
  const minE = Number.parseInt(epochSlider.min, 10);
  const maxE = Number.parseInt(epochSlider.max, 10);
  const e = Math.min(Math.max(epoch, minE), maxE);
  playbackEpoch = e;
  epochSlider.value = String(e);
  epochLabel.textContent = `#${e}`;
  renderEpochMetricsForEpoch(e);
  renderModelDiagram(
    currentConfig,
    currentRunMeta,
    metricToActiveTerm(metricSelect.value),
    getNearestRecordByEpoch(currentLogRecords, e),
  );
  renderCharts();
}

function startPlayback() {
  if (!currentLogRecords || currentLogRecords.length === 0) return;
  stopPlayback();
  playBtn.disabled = true;
  pauseBtn.disabled = false;
  const baseMs = 250;
  playbackTimer = window.setInterval(() => {
    const speed = Number.parseInt(speedSelect.value, 10) || 1;
    const cur = (playbackEpoch ?? Number.parseInt(epochSlider.value, 10)) || 0;
    const maxE = Number.parseInt(epochSlider.max, 10);
    const next = cur + Math.max(1, speed);
    if (next > maxE) {
      setEpoch(maxE);
      stopPlayback();
      if (isRecording) stopRecording();
      return;
    }
    setEpoch(next);
  }, baseMs);
}

function renderCharts() {
  if (!currentLogRecords || currentLogRecords.length === 0) {
    drawLine(curveCanvas, [], { title: metricSelect.value });
    drawMultiLine(lossCanvas, [], { title: "loss components" });
    return;
  }
  const cursorEpoch = (playbackEpoch ?? Number.parseInt(epochSlider.value, 10)) || null;

  // main curve
  const pts = getMetric(currentLogRecords, metricSelect.value);
  drawLine(curveCanvas, pts, { title: metricSelect.value, cursorEpoch });

  // loss decomposition
  if (!currentConfig) {
    drawMultiLine(lossCanvas, [], { title: "loss components" });
    setLossStatus("config.txt missing; cannot compute weighted terms.", "error");
  } else {
    const series = computeLossSeries(currentLogRecords, currentConfig);
    drawMultiLine(lossCanvas, series, { title: "train loss components", cursorEpoch });
    const lambdaSmooth = cfgNum(currentConfig, "lambda_smooth", null);
    const lambdaTrans = cfgNum(currentConfig, "lambda_transition", null);
    const lambdaLle = cfgNum(currentConfig, "lambda_lle", null);
    setLossStatus(
      `Using λsmooth=${lambdaSmooth ?? "—"}, λtrans=${lambdaTrans ?? "—"}, λlle=${lambdaLle ?? "—"} (and per-epoch beta from run.log).`,
    );
  }

  if (isRecording) drawExportFrame();
}

function setRecordStatus(msg) {
  if (!recordStatus) return;
  recordStatus.textContent = msg;
}

function pickRecordingMimeType() {
  if (!window.MediaRecorder || !window.MediaRecorder.isTypeSupported) return "";
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  for (const t of candidates) {
    try {
      if (MediaRecorder.isTypeSupported(t)) return t;
    } catch {
      // ignore
    }
  }
  return "";
}

function startRecording() {
  if (!currentRunId) {
    setCurveStatus("Select a run before recording.", "error");
    return;
  }
  if (!currentLogRecords || currentLogRecords.length === 0) {
    setCurveStatus("No epoch records parsed from run.log; cannot record.", "error");
    return;
  }
  if (!curveCanvas?.captureStream) {
    setCurveStatus("Canvas captureStream() not supported in this browser.", "error");
    return;
  }
  if (!window.MediaRecorder) {
    setCurveStatus("MediaRecorder not supported in this browser.", "error");
    return;
  }

  ensureExportCanvas();
  drawExportFrame();

  const stream = exportCanvas.captureStream(30);
  const mimeType = pickRecordingMimeType();
  recordedChunks = [];
  isRecording = true;

  try {
    mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
  } catch (e) {
    isRecording = false;
    setCurveStatus(`Failed to start recorder: ${String(e?.message ?? e)}`, "error");
    return;
  }

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: mediaRecorder?.mimeType || "video/webm" });
    const url = URL.createObjectURL(blob);
    const safeMetric = (metricSelect?.value ?? "metric").replace(/[^\w.-]+/g, "_");
    const a = document.createElement("a");
    a.href = url;
    a.download = `${currentRunId}_${safeMetric}.webm`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1500);
    setRecordStatus("Saved .webm download.");
  };

  recordBtn.disabled = true;
  stopRecordBtn.disabled = false;
  setRecordStatus("Recording…");

  try {
    mediaRecorder.start(250);
  } catch (e) {
    isRecording = false;
    recordBtn.disabled = false;
    stopRecordBtn.disabled = true;
    setCurveStatus(`Recorder start() failed: ${String(e?.message ?? e)}`, "error");
    return;
  }

  if (!playbackTimer) startPlayback();
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  recordBtn.disabled = false;
  stopRecordBtn.disabled = true;
  setRecordStatus("Finalizing…");
  try {
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  } catch (e) {
    setRecordStatus(`Stop failed: ${String(e?.message ?? e)}`);
  }
}

function snapshotPng() {
  if (!currentRunId) {
    setCurveStatus("Select a run before snapshot.", "error");
    return;
  }
  ensureExportCanvas();
  drawExportFrame();
  exportCanvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const safeMetric = (metricSelect?.value ?? "metric").replace(/[^\w.-]+/g, "_");
    const a = document.createElement("a");
    a.href = url;
    a.download = `${currentRunId}_${safeMetric}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1500);
  }, "image/png");
}

async function loadAndRenderRun(runId) {
  currentRunId = runId;
  stopPlayback();
  stopRecording();
  setRecordStatus("");
  fileCache = new Map();
  currentRunJson = null;
  currentConfig = null;
  currentConfigText = null;
  currentRunLogText = null;
  currentLogRecords = null;
  currentRunMeta = null;
  selectedLayerKey = null;
  currentArch = null;
  playbackEpoch = null;

  epochMetrics.innerHTML = "";
  fileViewer.textContent = "(loading…)";
  if (modelDiagram) modelDiagram.textContent = "(loading…)";
  setCurveStatus("");
  setLossStatus("");

  setImgOrMissing(previewImg, `../neuroscience/runs/${runId}/preview.png`);
  setImgOrMissing(manifoldImg, `../neuroscience/runs/${runId}/latent_manifold_mds.png`);

  renderDetails(null, null);
  clearCanvas(curveCanvas);
  clearCanvas(lossCanvas);
  setCurveStatus("Loading run.json, config.txt, and run.log…");

  const runJsonP = fetch(`../neuroscience/runs/${runId}/run.json`, { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);
  const cfgTextP = fetch(`../neuroscience/runs/${runId}/config.txt`, { cache: "no-store" })
    .then((r) => (r.ok ? r.text() : null))
    .catch(() => null);
  const logP = fetch(`../neuroscience/runs/${runId}/run.log`, { cache: "no-store" })
    .then((r) => (r.ok ? r.text() : null))
    .catch(() => null);

  const [runJson, cfgText, logText] = await Promise.all([runJsonP, cfgTextP, logP]);
  const cfg = cfgText ? parseConfig(cfgText) : null;

  currentRunJson = runJson;
  currentConfig = cfg;
  currentConfigText = cfgText;
  currentRunLogText = logText;
  currentRunMeta = logText ? parseRunMeta(logText) : null;

  if (currentRunJson) fileCache.set("run.json", JSON.stringify(currentRunJson, null, 2));
  if (currentConfigText) fileCache.set("config.txt", currentConfigText);
  if (currentRunLogText !== null && currentRunLogText !== undefined) fileCache.set("run.log", currentRunLogText);

  renderDetails(currentRunJson, currentConfig);
  renderModelDiagram(currentConfig, currentRunMeta, metricToActiveTerm(metricSelect.value), null);

  if (!logText) {
    currentLogRecords = [];
    renderCharts();
    setCurveStatus("run.log not found (or could not be loaded).", "error");
    setSelectedTab(tabRunJson);
    await showFile("run.json");
    return;
  }

  currentLogRecords = parseRunLog(logText);
  if (currentLogRecords.length === 0) {
    setCurveStatus("run.log loaded but no epochs were parsed (format mismatch?).", "error");
  }

  // initialize epoch slider (default: last epoch)
  if (currentLogRecords.length > 0) {
    const epochs = currentLogRecords.map((r) => r.epoch);
    const minE = Math.min(...epochs);
    const maxE = Math.max(...epochs);
    epochSlider.min = String(minE);
    epochSlider.max = String(maxE);
    epochSlider.value = String(maxE);
    setEpoch(maxE);
  } else {
    epochSlider.min = "0";
    epochSlider.max = "0";
    epochSlider.value = "0";
    epochLabel.textContent = "—";
    renderCharts();
  }

  setCurveStatus(`Parsed ${currentLogRecords.length} epochs from run.log.`);

  setSelectedTab(tabRunJson);
  await showFile("run.json");
}

function attachEvents() {
  filterInput.addEventListener("input", applyFilter);
  reloadBtn.addEventListener("click", async () => {
    try {
      await loadRunsIndex();
    } catch (e) {
      setStatus(String(e?.message ?? e), "error");
    }
  });
  runSelect.addEventListener("change", async () => {
    const runId = runSelect.value;
    if (!runId) return;
    try {
      await loadAndRenderRun(runId);
    } catch (e) {
      setCurveStatus(String(e?.message ?? e), "error");
    }
  });
  playBtn.addEventListener("click", () => startPlayback());
  pauseBtn.addEventListener("click", () => stopPlayback());
  speedSelect.addEventListener("change", () => {
    if (playbackTimer) {
      startPlayback();
    }
  });
  epochSlider.addEventListener("input", () => {
    stopPlayback();
    const e = Number.parseInt(epochSlider.value, 10);
    if (Number.isFinite(e)) setEpoch(e);
  });
  metricSelect.addEventListener("change", () => {
    const cursorEpoch = (playbackEpoch ?? Number.parseInt(epochSlider.value, 10)) || null;
    const cursorRec = getNearestRecordByEpoch(currentLogRecords, cursorEpoch);
    renderModelDiagram(currentConfig, currentRunMeta, metricToActiveTerm(metricSelect.value), cursorRec);
    renderCharts();
  });
  window.addEventListener("resize", () => {
    renderCharts();
  });

  if (modelDiagram) {
    const handleLayerSelect = (target) => {
      const card = target?.closest?.(".layerCard");
      if (!card) return;
      const key = card.getAttribute("data-layer-key");
      if (!key) return;
      selectedLayerKey = selectedLayerKey === key ? null : key;
      const cursorEpoch = (playbackEpoch ?? Number.parseInt(epochSlider.value, 10)) || null;
      const cursorRec = getNearestRecordByEpoch(currentLogRecords, cursorEpoch);
      renderModelDiagram(currentConfig, currentRunMeta, metricToActiveTerm(metricSelect.value), cursorRec);
    };

    modelDiagram.addEventListener("click", (e) => handleLayerSelect(e.target));
    modelDiagram.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      handleLayerSelect(e.target);
      e.preventDefault();
    });
  }

  recordBtn.addEventListener("click", () => startRecording());
  stopRecordBtn.addEventListener("click", () => stopRecording());
  snapshotBtn.addEventListener("click", () => snapshotPng());

  tabRunJson.addEventListener("click", async () => {
    setSelectedTab(tabRunJson);
    await showFile("run.json");
  });
  tabConfig.addEventListener("click", async () => {
    setSelectedTab(tabConfig);
    await showFile("config.txt");
  });
  tabConfigOrig.addEventListener("click", async () => {
    setSelectedTab(tabConfigOrig);
    await showFile("config_original.txt");
  });
  tabRunLog.addEventListener("click", async () => {
    setSelectedTab(tabRunLog);
    await showFile("run.log");
  });
  tabGitDiff.addEventListener("click", async () => {
    setSelectedTab(tabGitDiff);
    await showFile("git_diff.patch");
  });
}

async function main() {
  attachEvents();
  try {
    await loadRunsIndex();
  } catch (e) {
    setStatus(
      `Failed to load runs index. Make sure you started a local server from the repo root. (${String(
        e?.message ?? e,
      )})`,
      "error",
    );
  }
}

main();
