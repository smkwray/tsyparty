/* Manifest-driven data loader for tsyparty static site.
 * Reads manifests first, then loads CSV/JSON artifacts as needed. */

const DATA_BASE = "../outputs";
const DERIVED_BASE = "../data/derived";
const INTERIM_BASE = "../data/interim";

const dataCache = {};

async function fetchJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to load ${url}: ${resp.status}`);
  return resp.json();
}

async function fetchCSV(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to load ${url}: ${resp.status}`);
  const text = await resp.text();
  return Papa.parse(text.trim(), { header: true, dynamicTyping: true, skipEmptyLines: true }).data;
}

async function loadManifest(pipeline) {
  const key = `manifest_${pipeline}`;
  if (dataCache[key]) return dataCache[key];
  try {
    const manifest = await fetchJSON(`${DATA_BASE}/${pipeline}/manifest.json`);
    dataCache[key] = manifest;
    return manifest;
  } catch {
    return null;
  }
}

async function resolvePreferredSimilarityPipeline() {
  const [enriched, base] = await Promise.all([
    loadManifest("similarity_enriched"),
    loadManifest("similarity"),
  ]);
  if (enriched && enriched.status === "ok") {
    return { pipeline: "similarity_enriched", manifest: enriched, fallbackManifest: base };
  }
  return { pipeline: "similarity", manifest: base, fallbackManifest: enriched };
}

async function loadPipelineFile(pipeline, filename) {
  const key = `${pipeline}/${filename}`;
  if (dataCache[key]) return dataCache[key];
  const url = `${DATA_BASE}/${pipeline}/${filename}`;
  let data;
  if (filename.endsWith(".json")) {
    data = await fetchJSON(url);
  } else if (filename.endsWith(".csv")) {
    data = await fetchCSV(url);
  } else {
    throw new Error(`Unknown file type: ${filename}`);
  }
  dataCache[key] = data;
  return data;
}

async function loadDerived(filename) {
  const key = `derived/${filename}`;
  if (dataCache[key]) return dataCache[key];
  const data = await fetchCSV(`${DERIVED_BASE}/${filename}`);
  dataCache[key] = data;
  return data;
}

async function loadInterim(filename) {
  const key = `interim/${filename}`;
  if (dataCache[key]) return dataCache[key];
  const data = await fetchCSV(`${INTERIM_BASE}/${filename}`);
  dataCache[key] = data;
  return data;
}

async function loadOutputFile(subdir, filename) {
  const key = `${subdir}/${filename}`;
  if (dataCache[key]) return dataCache[key];
  const url = `${DATA_BASE}/${subdir}/${filename}`;
  let data;
  if (filename.endsWith(".json")) {
    data = await fetchJSON(url);
  } else {
    data = await fetchCSV(url);
  }
  dataCache[key] = data;
  return data;
}

/* Load all files for a pipeline using its manifest */
async function loadPipeline(pipeline) {
  const manifest = await loadManifest(pipeline);
  if (!manifest) return { manifest: null, data: {} };
  if (manifest.status === "no_data") return { manifest, data: {} };

  const data = {};
  const files = manifest.files_written || [];
  await Promise.all(files.map(async (f) => {
    if (f === "manifest.json") return;
    try {
      data[f] = await loadPipelineFile(pipeline, f);
    } catch (e) {
      console.warn(`Could not load ${pipeline}/${f}:`, e.message);
    }
  }));
  return { manifest, data };
}

/* Format numbers nicely */
function fmt(n, decimals = 1) {
  if (n == null || isNaN(n)) return "—";
  if (Math.abs(n) >= 1e12) return (n / 1e12).toFixed(decimals) + "T";
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(decimals) + "B";
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(decimals) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(decimals) + "K";
  return Number(n).toFixed(decimals);
}

function pct(n, decimals = 1) {
  if (n == null || isNaN(n)) return "—";
  return (n * 100).toFixed(decimals) + "%";
}

function fmtDate(d) {
  if (!d) return "—";
  const date = new Date(d);
  return date.toLocaleDateString("en-US", { year: "numeric", month: "short" });
}

function metricLabel(metric) {
  const labels = {
    "partial_pearson": "Partial Correlation",
    "cosine": "Cosine Distance",
    "pearson": "Pearson Correlation",
    "spearman": "Spearman Correlation",
  };
  return labels[metric] || metric.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}
