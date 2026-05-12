const statusEl = document.getElementById("status");
const appInfo = window.vecdbApp;
const API_BASE_URL = "http://localhost:3001";

const queryInput = document.getElementById("query-input");
const queryButton = document.getElementById("query-button");
const uploadButton = document.getElementById("upload-button");
const thresholdSlider = document.getElementById("threshold-slider");
const thresholdValue = document.getElementById("threshold-value");
const resultsEl = document.getElementById("results");

const processRefreshButton = document.getElementById("process-refresh-button");
const processStatus = document.getElementById("process-status");
const processSummary = document.getElementById("process-summary");
const processStepper = document.getElementById("process-stepper");
const processDetail = document.getElementById("process-detail");
const tokenList = document.getElementById("token-list");
const tokenDetail = document.getElementById("token-detail");
const vectorPreview = document.getElementById("vector-preview");
const vectorStats = document.getElementById("vector-stats");

const mapMethod = document.getElementById("map-method");
const mapLimit = document.getElementById("map-limit");
const mapButton = document.getElementById("map-button");
const mapStatus = document.getElementById("map-status");
const mapCanvas = document.getElementById("map-canvas");
const mapTooltip = document.getElementById("map-tooltip");
const clusterLegend = document.getElementById("cluster-legend");

const cosineCanvas = document.getElementById("cosine-canvas");
const cosineDescription = document.getElementById("cosine-description");
const cosineMetrics = document.getElementById("cosine-metrics");
const cosineFormula = document.getElementById("cosine-formula");
const cosineControlsA = document.getElementById("cosine-controls-a");
const cosineControlsB = document.getElementById("cosine-controls-b");
const cosinePresetButtons = Array.from(
  document.querySelectorAll("[data-cosine-preset]")
);

const viewTabs = Array.from(document.querySelectorAll("[data-tab-target]"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));

const CLUSTER_COLORS = [
  "#2563eb",
  "#dc2626",
  "#16a34a",
  "#ca8a04",
  "#7c3aed",
  "#0891b2",
  "#ea580c",
  "#db2777",
  "#4f46e5",
  "#65a30d",
];

const MAX_CLUSTER_COUNT = CLUSTER_COLORS.length;
const PROCESS_STEP_IDS = ["tokenize", "tensorize", "encode", "normalize"];
const COSINE_AXES = [
  { key: "x", label: "X" },
  { key: "y", label: "Y" },
  { key: "z", label: "Z" },
];
const COSINE_LIMIT = 3;
const COSINE_PRESETS = {
  identical: {
    a: { x: 1.6, y: 0.9, z: -0.5 },
    b: { x: 1.6, y: 0.9, z: -0.5 },
  },
  orthogonal: {
    a: { x: 1, y: 0, z: 0 },
    b: { x: 0, y: 1, z: 0 },
  },
  opposite: {
    a: { x: 1.2, y: -0.8, z: 0.7 },
    b: { x: -1.2, y: 0.8, z: -0.7 },
  },
  nearby: {
    a: { x: 1.8, y: 0.8, z: -0.2 },
    b: { x: 1.2, y: 1.1, z: 0.3 },
  },
};

let lastQueryText = "";
let lastQueryVector = null;
let lastQueryProcess = null;
let activeProcessStepId = PROCESS_STEP_IDS[0];
let selectedProcessTokenIndex = 0;
let mapPoints = [];
let mapData = [];
let returnedResultImageIds = new Set();
let cosineVectors = JSON.parse(JSON.stringify(COSINE_PRESETS.nearby));
const cosineControlRefs = {
  a: {},
  b: {},
};

function setStatus(message) {
  if (statusEl) {
    statusEl.textContent = message;
  }
}

function setUploadButtonState(isUploading, total = 0) {
  if (!uploadButton) {
    return;
  }

  uploadButton.disabled = isUploading;
  uploadButton.classList.toggle("is-uploading", isUploading);

  if (isUploading) {
    uploadButton.innerHTML = `
      <span class="upload-spinner" aria-hidden="true"></span>
      <span class="upload-label">Uploading ${total} image${total === 1 ? "" : "s"}</span>
    `;
    return;
  }

  uploadButton.innerHTML = '<span class="upload-label">Upload Images</span>';
}

function clearResults() {
  if (resultsEl) {
    resultsEl.replaceChildren();
  }
}

function setMapStatus(message) {
  if (mapStatus) {
    mapStatus.textContent = message;
  }
}

function setProcessStatus(message) {
  if (processStatus) {
    processStatus.textContent = message;
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatThreshold(value) {
  return Number(value).toFixed(2);
}

function formatNumber(value, digits = 3) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "Unavailable";
}

function formatSignedNumber(value, digits = 3) {
  if (!Number.isFinite(Number(value))) {
    return "Unavailable";
  }

  const numericValue = Number(value);
  return `${numericValue >= 0 ? "+" : ""}${numericValue.toFixed(digits)}`;
}

function formatPercent(value, digits = 1) {
  return Number.isFinite(Number(value))
    ? `${(Number(value) * 100).toFixed(digits)}%`
    : "Unavailable";
}

function formatShape(shape) {
  if (!Array.isArray(shape) || !shape.length) {
    return "Unavailable";
  }

  return `[${shape.join(" x ")}]`;
}

function truncateText(text, maxLength = 38) {
  if (typeof text !== "string" || text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength - 1)}...`;
}

function formatTokenText(text) {
  if (typeof text !== "string") {
    return "<?>"; 
  }

  const visible = text.replace(/ /g, "␠").replace(/\n/g, "↵");
  return visible || "(blank)";
}

function getSimilarityThreshold() {
  const parsed = Number(thresholdSlider?.value);
  return Number.isFinite(parsed) ? parsed : 0.35;
}

function updateThresholdValue() {
  const threshold = getSimilarityThreshold();

  if (thresholdValue) {
    thresholdValue.textContent = formatThreshold(threshold);
  }
}

function formatSearchSummary(result, noun = "images") {
  const total = result.total || result.matches?.length || 0;
  const threshold = formatThreshold(result.within ?? getSimilarityThreshold());
  const fallback = result.thresholdFallback
    ? " No close matches were found, so the backend retried without the threshold."
    : "";

  return `${total} ${noun} ranked by similarity. Distance threshold: ${threshold}.${fallback}`;
}

function getMatchImage(match) {
  return match.image || match;
}

function formatSimilarity(value) {
  if (typeof value !== "number") {
    return "";
  }

  return `${Math.round(value * 100)}% match`;
}

function normalizeSimilarityValue(point) {
  if (typeof point?.similarity === "number" && Number.isFinite(point.similarity)) {
    return clamp(point.similarity, 0, 1);
  }

  if (typeof point?.distance === "number" && Number.isFinite(point.distance)) {
    return clamp(1 - point.distance, 0, 1);
  }

  return null;
}

function formatEmbeddingVector(vector, previewLength = 12) {
  if (!Array.isArray(vector) || !vector.length) {
    return "No vector data available.";
  }

  const preview = vector
    .slice(0, previewLength)
    .map((value) => (typeof value === "number" ? value.toFixed(6) : String(value)))
    .join(", ");
  const remaining = vector.length - previewLength;

  return `[${preview}${remaining > 0 ? `, ... (+${remaining} more)` : ""}]`;
}

function formatEmbeddingSummary(embedding, index) {
  const parts = [`Embedding ${index + 1}`];

  if (embedding?.modelUsed) {
    parts.push(embedding.modelUsed);
  }

  if (typeof embedding?.dimension === "number") {
    parts.push(`${embedding.dimension} dims`);
  }

  return parts.join(" • ");
}

function ensureCanvasSize(canvasElement) {
  if (!canvasElement) {
    return false;
  }

  const rect = canvasElement.getBoundingClientRect();
  if (!rect.width || !rect.height) {
    return false;
  }

  const dpr = window.devicePixelRatio || 1;
  canvasElement.width = Math.max(1, Math.floor(rect.width * dpr));
  canvasElement.height = Math.max(1, Math.floor(rect.height * dpr));
  return true;
}

function squaredDistance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function chooseClusterCount(points) {
  if (points.length <= 1) {
    return points.length;
  }

  return clamp(Math.round(Math.sqrt(points.length / 12)), 2, MAX_CLUSTER_COUNT);
}

function nearestCentroidIndex(point, centroids) {
  let bestIndex = 0;
  let bestDistance = Infinity;

  centroids.forEach((centroid, index) => {
    const distance = squaredDistance(point, centroid);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });

  return bestIndex;
}

function initializeCentroids(points, clusterCount) {
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const centroids = [sorted[Math.floor(sorted.length / 2)]];

  while (centroids.length < clusterCount) {
    const next = sorted.reduce(
      (best, point) => {
        const distance = Math.min(
          ...centroids.map((centroid) => squaredDistance(point, centroid))
        );

        return distance > best.distance ? { point, distance } : best;
      },
      { point: sorted[0], distance: -1 }
    );

    centroids.push(next.point);
  }

  return centroids.map((point) => ({ x: point.x, y: point.y }));
}

function clusterMapPoints(points) {
  const matchPoints = points.filter((point) => point.kind !== "query");
  const clusterCount = chooseClusterCount(matchPoints);

  if (!clusterCount) {
    return {
      points,
      clusters: [],
    };
  }

  let centroids = initializeCentroids(matchPoints, clusterCount);
  let assignments = new Map();

  for (let iteration = 0; iteration < 18; iteration += 1) {
    assignments = new Map();
    const totals = centroids.map(() => ({ x: 0, y: 0, count: 0 }));

    matchPoints.forEach((point) => {
      const clusterIndex = nearestCentroidIndex(point, centroids);
      assignments.set(point.plotIndex, clusterIndex);
      totals[clusterIndex].x += point.x;
      totals[clusterIndex].y += point.y;
      totals[clusterIndex].count += 1;
    });

    centroids = centroids.map((centroid, index) => {
      const total = totals[index];
      if (!total.count) {
        return centroid;
      }

      return {
        x: total.x / total.count,
        y: total.y / total.count,
      };
    });
  }

  const clusters = centroids.map((_, index) => ({
    id: index + 1,
    color: CLUSTER_COLORS[index % CLUSTER_COLORS.length],
    size: 0,
  }));

  const clusteredPoints = points.map((point) => {
    if (point.kind === "query") {
      return {
        ...point,
        clusterId: null,
        clusterColor: "#111c33",
      };
    }

    const clusterIndex = assignments.get(point.plotIndex) ?? 0;
    clusters[clusterIndex].size += 1;

    return {
      ...point,
      clusterId: clusterIndex + 1,
      clusterColor: clusters[clusterIndex].color,
    };
  });

  return {
    points: clusteredPoints,
    clusters: clusters.filter((cluster) => cluster.size > 0),
  };
}

function updateClusterLegend(clusters) {
  if (!clusterLegend) {
    return;
  }

  clusterLegend.replaceChildren();

  if (!clusters.length) {
    const placeholder = document.createElement("span");
    placeholder.className = "cluster-placeholder";
    placeholder.textContent = "Clusters appear after map generation.";
    clusterLegend.appendChild(placeholder);
    return;
  }

  clusters.forEach((cluster) => {
    const item = document.createElement("span");
    item.className = "cluster-legend-item";

    const swatch = document.createElement("span");
    swatch.className = "cluster-swatch";
    swatch.style.backgroundColor = cluster.color;

    const label = document.createElement("span");
    label.textContent = `Cluster ${cluster.id} (${cluster.size})`;

    item.append(swatch, label);
    clusterLegend.appendChild(item);
  });

  if (returnedResultImageIds.size) {
    const resultItem = document.createElement("span");
    resultItem.className = "cluster-legend-item";

    const resultSwatch = document.createElement("span");
    resultSwatch.className = "returned-result-swatch";

    const resultLabel = document.createElement("span");
    resultLabel.textContent = "Returned result";

    resultItem.append(resultSwatch, resultLabel);
    clusterLegend.appendChild(resultItem);
  }
}

function drawClusterPoint(ctx, point, dpr) {
  const radius = 5;

  ctx.fillStyle = point.clusterColor;
  ctx.beginPath();
  ctx.arc(point.screenX * dpr, point.screenY * dpr, radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(13, 26, 53, 0.28)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(point.screenX * dpr, point.screenY * dpr, radius + 0.5, 0, Math.PI * 2);
  ctx.stroke();
}

function drawReturnedResultPoint(ctx, point, dpr) {
  const x = point.screenX * dpr;
  const y = point.screenY * dpr;

  ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
  ctx.beginPath();
  ctx.arc(x, y, 13, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "#111c33";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(x, y, 11, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = point.clusterColor;
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.stroke();
}

function drawQueryPoint(ctx, point, dpr) {
  const radius = 8;

  ctx.fillStyle = "#111c33";
  ctx.beginPath();
  ctx.arc(point.screenX * dpr, point.screenY * dpr, radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(point.screenX * dpr, point.screenY * dpr, radius + 2, 0, Math.PI * 2);
  ctx.stroke();
}

function renderMap(points = mapData) {
  if (!mapCanvas) {
    return;
  }

  const currentPoints = Array.isArray(points) ? points : [];
  mapData = currentPoints;

  if (!ensureCanvasSize(mapCanvas)) {
    return;
  }

  const ctx = mapCanvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const width = mapCanvas.width;
  const height = mapCanvas.height;
  ctx.clearRect(0, 0, width, height);

  if (!currentPoints.length) {
    mapPoints = [];
    updateClusterLegend([]);
    return;
  }

  const xs = currentPoints.map((point) => point.x);
  const ys = currentPoints.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const padding = 32;
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const scaleX = (value) =>
    ((value - minX) / spanX) * (width - padding * 2) + padding;
  const scaleY = (value) =>
    height - (((value - minY) / spanY) * (height - padding * 2) + padding);

  ctx.fillStyle = "#e3ecff";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(13, 26, 53, 0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(padding, padding, width - padding * 2, height - padding * 2);
  ctx.stroke();

  const axisX = minX <= 0 && maxX >= 0 ? scaleX(0) : padding;
  const axisY = minY <= 0 && maxY >= 0 ? scaleY(0) : height - padding;

  ctx.strokeStyle = "rgba(13, 26, 53, 0.25)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padding, axisY);
  ctx.lineTo(width - padding, axisY);
  ctx.moveTo(axisX, padding);
  ctx.lineTo(axisX, height - padding);
  ctx.stroke();

  ctx.fillStyle = "rgba(13, 26, 53, 0.6)";
  ctx.font = "12px 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
  const tickCount = 5;
  for (let index = 0; index <= tickCount; index += 1) {
    const t = index / tickCount;
    const xValue = minX + t * spanX;
    const yValue = minY + t * spanY;
    const xPos = padding + t * (width - padding * 2);
    const yPos = height - (padding + t * (height - padding * 2));

    ctx.strokeStyle = "rgba(13, 26, 53, 0.18)";
    ctx.beginPath();
    ctx.moveTo(xPos, axisY - 4);
    ctx.lineTo(xPos, axisY + 4);
    ctx.moveTo(axisX - 4, yPos);
    ctx.lineTo(axisX + 4, yPos);
    ctx.stroke();

    ctx.fillText(xValue.toFixed(2), xPos - 12, axisY + 18);
    ctx.fillText(yValue.toFixed(2), axisX + 8, yPos + 4);
  }

  const dpr = window.devicePixelRatio || 1;
  const plotted = currentPoints.map((point, index) => {
    const similarity = normalizeSimilarityValue(point);

    return {
      ...point,
      plotIndex: index,
      isReturnedResult:
        point.kind !== "query" && returnedResultImageIds.has(point.id),
      similarity,
      screenX: scaleX(point.x) / dpr,
      screenY: scaleY(point.y) / dpr,
    };
  });

  const clustered = clusterMapPoints(plotted);
  updateClusterLegend(clustered.clusters);

  clustered.points
    .filter((point) => point.kind !== "query" && !point.isReturnedResult)
    .forEach((point) => drawClusterPoint(ctx, point, dpr));

  clustered.points
    .filter((point) => point.isReturnedResult)
    .forEach((point) => drawReturnedResultPoint(ctx, point, dpr));

  clustered.points
    .filter((point) => point.kind === "query")
    .forEach((point) => drawQueryPoint(ctx, point, dpr));

  mapPoints = clustered.points;
}

function showTooltip(point, event) {
  if (!mapTooltip || !point) {
    return;
  }

  const label =
    point.kind === "query"
      ? "Query"
      : point.isReturnedResult
        ? `Returned result: ${point.label || "Result"}`
        : point.label || "Cluster point";
  const similarity =
    typeof point.similarity === "number"
      ? `${Math.round(point.similarity * 100)}% match`
      : "Similarity unavailable";
  const cluster = point.clusterId ? `Cluster ${point.clusterId}` : "Query point";

  mapTooltip.textContent = `${label} - ${cluster} - ${similarity}`;
  mapTooltip.style.left = `${event.offsetX}px`;
  mapTooltip.style.top = `${event.offsetY}px`;
  mapTooltip.style.opacity = "1";
}

function hideTooltip() {
  if (mapTooltip) {
    mapTooltip.style.opacity = "0";
  }
}

function handleMapHover(event) {
  if (!mapPoints.length) {
    return;
  }

  const hit = mapPoints.find((point) => {
    const dx = event.offsetX - point.screenX;
    const dy = event.offsetY - point.screenY;
    const hitRadius = point.isReturnedResult ? 16 : 10;
    return Math.sqrt(dx * dx + dy * dy) < hitRadius;
  });

  if (hit) {
    showTooltip(hit, event);
  } else {
    hideTooltip();
  }
}

function createEmbeddingsDropdown(embeddings = []) {
  const dropdown = document.createElement("details");
  dropdown.className = "embedding-dropdown";

  const summary = document.createElement("summary");
  summary.textContent = `Embeddings (${embeddings.length})`;
  dropdown.appendChild(summary);

  const list = document.createElement("div");
  list.className = "embedding-list";

  embeddings.forEach((embedding, index) => {
    const item = document.createElement("section");
    item.className = "embedding-item";

    const heading = document.createElement("p");
    heading.className = "embedding-title";
    heading.textContent = formatEmbeddingSummary(embedding, index);

    const meta = document.createElement("p");
    meta.className = "embedding-meta";
    meta.textContent = embedding?.generatedAt
      ? `Generated ${new Date(embedding.generatedAt).toLocaleString()}`
      : "Generated time unavailable";

    const vector = document.createElement("pre");
    vector.className = "embedding-vector";
    vector.textContent = formatEmbeddingVector(embedding?.vector);

    item.append(heading, meta, vector);
    list.appendChild(item);
  });

  dropdown.appendChild(list);
  return dropdown;
}

function renderResults(matches = []) {
  clearResults();
  returnedResultImageIds = new Set();

  if (!resultsEl) {
    return;
  }

  if (!matches.length) {
    const empty = document.createElement("p");
    empty.className = "empty-results";
    empty.textContent = "No matching images found.";
    resultsEl.appendChild(empty);
    return;
  }

  const fragment = document.createDocumentFragment();

  matches.forEach((match) => {
    const image = getMatchImage(match);
    if (!image?.imageUrl) {
      return;
    }

    if (image.id) {
      returnedResultImageIds.add(image.id);
    }

    const card = document.createElement("article");
    card.className = "result-card";

    const img = document.createElement("img");
    img.src = image.imageUrl;
    img.alt = image.description || "VisionDB result";
    img.loading = "lazy";

    const details = document.createElement("div");
    details.className = "result-details";

    const title = document.createElement("h2");
    title.textContent = image.description || "Untitled image";

    const score = document.createElement("p");
    score.className = "result-score";
    score.textContent = formatSimilarity(match.similarity);

    const embeddings = Array.isArray(image.embeddings_on_image)
      ? image.embeddings_on_image
      : [];

    details.append(title, score);

    if (embeddings.length) {
      details.appendChild(createEmbeddingsDropdown(embeddings));
    }

    card.append(img, details);
    fragment.appendChild(card);
  });

  resultsEl.appendChild(fragment);
}

async function parseJsonResponse(response) {
  const bodyText = await response.text();
  let result = {};

  if (bodyText) {
    try {
      result = JSON.parse(bodyText);
    } catch (_error) {
      result = { error: bodyText };
    }
  }

  if (!response.ok) {
    throw new Error(result.error || "Request failed.");
  }

  return result;
}

function createPlaceholderParagraph(text) {
  const placeholder = document.createElement("p");
  placeholder.className = "process-placeholder";
  placeholder.textContent = text;
  return placeholder;
}

function getProcessStepDescriptors(process) {
  if (!process) {
    return [];
  }

  const tokenization = process.tokenization || {};
  const tensors = process.tensors || {};
  const embedding = process.embedding || {};

  return [
    {
      id: "tokenize",
      title: "1. Tokenize query",
      meta: `${tokenization.totalTokens || 0} tokens`,
      description: `The tokenizer split "${truncateText(
        process.queryText || "",
        64
      )}" into ${tokenization.totalTokens || 0} pieces, including ${
        tokenization.specialTokenCount || 0
      } special token${tokenization.specialTokenCount === 1 ? "" : "s"}.`,
      stats: [
        { label: "Characters", value: String((process.queryText || "").length) },
        { label: "Tokens", value: String(tokenization.totalTokens || 0) },
        {
          label: "Special tokens",
          value: String(tokenization.specialTokenCount || 0),
        },
        {
          label: "Truncated",
          value:
            tokenization.truncatedTokenCount > 0
              ? `${tokenization.truncatedTokenCount} removed`
              : "No",
        },
      ],
    },
    {
      id: "tensorize",
      title: "2. Build tensors",
      meta: formatShape(tensors.inputIdsShape),
      description:
        "Token IDs and the attention mask are packed into model-ready tensors. These shapes show exactly what the text encoder received.",
      stats: [
        { label: "Input IDs", value: formatShape(tensors.inputIdsShape) },
        {
          label: "Attention mask",
          value: formatShape(tensors.attentionMaskShape),
        },
        {
          label: "Mask-on tokens",
          value: String(tokenization.attendedTokenCount || 0),
        },
        {
          label: "Padding used",
          value: tokenization.hasPadding ? "Yes" : "No",
        },
      ],
    },
    {
      id: "encode",
      title: "3. Encode meaning",
      meta: process.modelUsed || "Model",
      description:
        "Jina CLIP's text encoder maps the token sequence into the same shared space used by your image embeddings, which is why text-to-image comparison works later on.",
      stats: [
        { label: "Model", value: process.modelUsed || "Unavailable" },
        {
          label: "Embedding source",
          value: tensors.embeddingSource || "Unavailable",
        },
        {
          label: "Hidden state",
          value: formatShape(tensors.hiddenStateShape),
        },
        {
          label: "Embedding tensor",
          value: formatShape(tensors.embeddingShape),
        },
      ],
    },
    {
      id: "normalize",
      title: "4. Normalize vector",
      meta: `${embedding.dimension || 0} dims`,
      description:
        "The final query vector is normalized so cosine similarity compares direction cleanly, not just raw scale. That makes similarity scores easier to interpret.",
      stats: [
        { label: "Dimensions", value: String(embedding.dimension || 0) },
        { label: "Vector norm", value: formatNumber(embedding.magnitude, 4) },
        { label: "Mean value", value: formatSignedNumber(embedding.mean, 4) },
        {
          label: "Value range",
          value: `${formatSignedNumber(embedding.min, 3)} to ${formatSignedNumber(
            embedding.max,
            3
          )}`,
        },
      ],
    },
  ];
}

function renderProcessSummary(process) {
  if (!processSummary) {
    return;
  }

  processSummary.replaceChildren();

  if (!process) {
    processSummary.appendChild(createPlaceholderParagraph("No query analyzed yet."));
    return;
  }

  const cards = [
    {
      value: truncateText(process.queryText || "", 26),
      label: "Current query",
    },
    {
      value: String(process.tokenization?.totalTokens || 0),
      label: "Tokens in sequence",
    },
    {
      value: `${process.embedding?.dimension || 0}D`,
      label: "Embedding size",
    },
    {
      value: formatNumber(process.embedding?.magnitude, 4),
      label: "Normalized vector norm",
    },
  ];

  cards.forEach((card) => {
    const article = document.createElement("article");
    article.className = "process-summary-card";
    article.title = card.value;

    const value = document.createElement("strong");
    value.textContent = card.value;

    const label = document.createElement("span");
    label.textContent = card.label;

    article.append(value, label);
    processSummary.appendChild(article);
  });
}

function renderProcessStepButtons(process) {
  if (!processStepper) {
    return;
  }

  processStepper.replaceChildren();

  const steps = getProcessStepDescriptors(process);
  if (!steps.length) {
    processStepper.appendChild(
      createPlaceholderParagraph("Step controls appear after a query is analyzed.")
    );
    return;
  }

  steps.forEach((step) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "process-step-button";
    button.classList.toggle("is-active", activeProcessStepId === step.id);
    button.dataset.processStep = step.id;

    const title = document.createElement("span");
    title.className = "process-step-title";
    title.textContent = step.title;

    const meta = document.createElement("span");
    meta.className = "process-step-meta";
    meta.textContent = step.meta;

    button.append(title, meta);
    button.addEventListener("click", () => {
      activeProcessStepId = step.id;
      renderProcessStepButtons(lastQueryProcess);
      renderProcessDetail(lastQueryProcess);
    });

    processStepper.appendChild(button);
  });
}

function renderProcessDetail(process) {
  if (!processDetail) {
    return;
  }

  processDetail.replaceChildren();

  const steps = getProcessStepDescriptors(process);
  const step =
    steps.find((item) => item.id === activeProcessStepId) || steps[0];

  if (!step) {
    processDetail.appendChild(
      createPlaceholderParagraph("Choose a step after analyzing a query.")
    );
    return;
  }

  activeProcessStepId = step.id;

  const header = document.createElement("div");
  header.className = "process-detail-header";

  const title = document.createElement("h3");
  title.textContent = step.title;

  const description = document.createElement("p");
  description.textContent = step.description;

  header.append(title, description);

  const stats = document.createElement("div");
  stats.className = "process-detail-stats";

  step.stats.forEach((item) => {
    const card = document.createElement("article");
    card.className = "process-stat-card";

    const value = document.createElement("strong");
    value.textContent = item.value;

    const label = document.createElement("span");
    label.textContent = item.label;

    card.append(value, label);
    stats.appendChild(card);
  });

  processDetail.append(header, stats);
}

function renderTokenDetail(process) {
  if (!tokenDetail) {
    return;
  }

  tokenDetail.replaceChildren();

  const tokens = process?.tokenization?.tokens || [];
  const token = tokens[selectedProcessTokenIndex];

  if (!token) {
    tokenDetail.appendChild(
      createPlaceholderParagraph("Token details appear after you select a token.")
    );
    return;
  }

  const tokenBadge = document.createElement("div");
  tokenBadge.className = "token-detail-token";
  tokenBadge.textContent = formatTokenText(token.text);

  const subtitle = document.createElement("p");
  subtitle.className = "process-note";
  subtitle.textContent = token.isSpecial
    ? "Special tokens help mark sequence boundaries or model-specific structure."
    : "Regular token emitted by the tokenizer for this query. This is diagnostic metadata, not a token-weight score.";

  const grid = document.createElement("div");
  grid.className = "token-detail-grid";

  const items = [
    { label: "Token index", value: String(token.index) },
    { label: "Token ID", value: String(token.id) },
    { label: "Mask on", value: token.attended ? "Yes" : "No" },
    { label: "Special token", value: token.isSpecial ? "Yes" : "No" },
  ];

  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "token-detail-item";

    const label = document.createElement("strong");
    label.textContent = item.label;

    const value = document.createElement("span");
    value.textContent = item.value;

    card.append(label, value);
    grid.appendChild(card);
  });

  tokenDetail.append(tokenBadge, subtitle, grid);
}

function renderTokenList(process) {
  if (!tokenList) {
    return;
  }

  tokenList.replaceChildren();

  const tokens = process?.tokenization?.tokens || [];
  if (!tokens.length) {
    tokenList.appendChild(
      createPlaceholderParagraph("Tokens appear here after analyzing a text query.")
    );
    renderTokenDetail(process);
    return;
  }

  selectedProcessTokenIndex = clamp(selectedProcessTokenIndex, 0, tokens.length - 1);

  tokens.forEach((token) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "token-chip";
    button.classList.toggle("is-special", token.isSpecial);
    button.classList.toggle("is-selected", token.index === selectedProcessTokenIndex);

    const index = document.createElement("span");
    index.className = "token-index";
    index.textContent = `#${token.index}`;

    const text = document.createElement("span");
    text.className = "token-text";
    text.textContent = formatTokenText(token.text);

    button.title = `Token ${token.index} • id ${token.id}`;
    button.append(index, text);
    button.addEventListener("click", () => {
      selectedProcessTokenIndex = token.index;
      renderTokenList(process);
      renderTokenDetail(process);
    });

    tokenList.appendChild(button);
  });

  renderTokenDetail(process);
}

function renderVectorPreview(process) {
  if (!vectorPreview || !vectorStats) {
    return;
  }

  vectorPreview.replaceChildren();
  vectorStats.replaceChildren();

  const embedding = process?.embedding;
  if (!embedding?.preview?.length) {
    vectorPreview.appendChild(
      createPlaceholderParagraph("Vector preview appears after a query is analyzed.")
    );
    vectorStats.appendChild(
      createPlaceholderParagraph("Vector statistics appear after a query is analyzed.")
    );
    return;
  }

  const previewValues = embedding.preview;
  const maxAbs = Math.max(...previewValues.map((value) => Math.abs(value))) || 1;

  previewValues.forEach((value, index) => {
    const item = document.createElement("div");
    item.className = "vector-bar";

    const fill = document.createElement("div");
    fill.className = "vector-bar-fill";
    fill.classList.toggle("is-negative", value < 0);
    fill.style.height = `${Math.max(12, (Math.abs(value) / maxAbs) * 150)}px`;
    fill.title = `Dimension ${index}: ${formatSignedNumber(value, 4)}`;

    const label = document.createElement("span");
    label.className = "vector-bar-label";
    label.textContent = index;

    item.append(fill, label);
    vectorPreview.appendChild(item);
  });

  const statCards = [
    { value: formatSignedNumber(embedding.mean, 4), label: "Mean value" },
    {
      value: `${embedding.positiveCount || 0}/${embedding.negativeCount || 0}`,
      label: "Positive / negative dims",
    },
    {
      value: `${formatSignedNumber(embedding.min, 3)} to ${formatSignedNumber(
        embedding.max,
        3
      )}`,
      label: "Min / max range",
    },
  ];

  statCards.forEach((cardInfo) => {
    const card = document.createElement("article");
    card.className = "process-stat-card";

    const value = document.createElement("strong");
    value.textContent = cardInfo.value;

    const label = document.createElement("span");
    label.textContent = cardInfo.label;

    card.append(value, label);
    vectorStats.appendChild(card);
  });

  const topDimensionsCard = document.createElement("article");
  topDimensionsCard.className = "process-stat-card";
  topDimensionsCard.style.gridColumn = "1 / -1";

  const heading = document.createElement("strong");
  heading.textContent = "Top dimensions";

  const label = document.createElement("span");
  label.textContent = "Largest values by absolute magnitude in the current query vector.";

  const pills = document.createElement("div");
  pills.className = "top-dimensions";

  (embedding.topDimensions || []).forEach((dimension) => {
    const pill = document.createElement("span");
    pill.className = "dimension-pill";
    pill.textContent = `d${dimension.index}: ${formatSignedNumber(dimension.value, 4)}`;
    pills.appendChild(pill);
  });

  topDimensionsCard.append(heading, label, pills);
  vectorStats.appendChild(topDimensionsCard);
}

function renderProcessInspector(process) {
  renderProcessSummary(process);
  renderProcessStepButtons(process);
  renderProcessDetail(process);
  renderTokenList(process);
  renderVectorPreview(process);
}

function storeProcessResponse(result) {
  const queryEmbedding = result?.queryEmbedding;

  if (queryEmbedding?.queryText) {
    lastQueryText = queryEmbedding.queryText;
  }

  if (Array.isArray(queryEmbedding?.vector)) {
    lastQueryVector = queryEmbedding.vector;
  }

  if (result?.process) {
    lastQueryProcess = result.process;
    selectedProcessTokenIndex = 0;
    renderProcessInspector(lastQueryProcess);
    setProcessStatus(
      `Inspected ${lastQueryProcess.tokenization?.totalTokens || 0} tokens and a ${
        lastQueryProcess.embedding?.dimension || 0
      }D embedding.`
    );
  }
}

async function inspectCurrentQuery(queryText = queryInput?.value || lastQueryText) {
  const cleaned = typeof queryText === "string" ? queryText.trim() : "";

  if (!cleaned) {
    setProcessStatus("Enter a text query before inspecting tokenization and vector-generation diagnostics.");
    return;
  }

  setProcessStatus("Inspecting tokenization and vector-generation diagnostics...");

  try {
    const response = await fetch(`${API_BASE_URL}/images/process/text`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ queryText: cleaned }),
    });

    const result = await parseJsonResponse(response);
    storeProcessResponse(result);
  } catch (error) {
    console.error("Process inspection error:", error);
    setProcessStatus(error.message || "Unable to inspect this query.");
  }
}

function dotProduct(a, b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function magnitude(vector) {
  return Math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

function cosineSimilarity(a, b) {
  const magnitudeA = magnitude(a);
  const magnitudeB = magnitude(b);

  if (!magnitudeA || !magnitudeB) {
    return null;
  }

  return clamp(dotProduct(a, b) / (magnitudeA * magnitudeB), -1, 1);
}

function describeCosineRelationship(value) {
  if (value === null) {
    return "Cosine similarity is undefined when one of the vectors has zero magnitude.";
  }

  if (value > 0.95) {
    return "The vectors point in almost the same direction.";
  }

  if (value > 0.5) {
    return "The vectors share a broadly similar direction.";
  }

  if (value > 0.1) {
    return "The vectors are only mildly aligned.";
  }

  if (value > -0.1) {
    return "The vectors are close to orthogonal, so they share very little directional overlap.";
  }

  if (value > -0.6) {
    return "The vectors trend away from each other.";
  }

  return "The vectors point in strongly opposite directions.";
}

function project3DPoint(point, width, height, scale) {
  const centerX = width / 2;
  const centerY = height / 2 + scale * 0.2;
  const projectedX = (point.x - point.z * 0.58) * scale;
  const projectedY = (point.y + point.x * 0.12 + point.z * 0.34) * scale;

  return {
    x: centerX + projectedX,
    y: centerY - projectedY,
  };
}

function drawArrow(ctx, from, to, color, width, label, dpr) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();

  const angle = Math.atan2(to.y - from.y, to.x - from.x);
  const headLength = 10 * dpr;

  ctx.beginPath();
  ctx.moveTo(to.x, to.y);
  ctx.lineTo(
    to.x - headLength * Math.cos(angle - Math.PI / 7),
    to.y - headLength * Math.sin(angle - Math.PI / 7)
  );
  ctx.lineTo(
    to.x - headLength * Math.cos(angle + Math.PI / 7),
    to.y - headLength * Math.sin(angle + Math.PI / 7)
  );
  ctx.closePath();
  ctx.fill();

  ctx.beginPath();
  ctx.arc(to.x, to.y, 5 * dpr, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = color;
  ctx.font = `${13 * dpr}px "Segoe UI", Tahoma, Geneva, Verdana, sans-serif`;
  ctx.fillText(label, to.x + 8 * dpr, to.y - 8 * dpr);
}

function renderCosineMetricsDisplay(metrics) {
  if (!cosineMetrics) {
    return;
  }

  cosineMetrics.replaceChildren();

  const cards = [
    {
      value:
        metrics.cosine === null ? "Undefined" : formatNumber(metrics.cosine, 3),
      label: "Cosine similarity",
    },
    {
      value:
        metrics.angleDegrees === null
          ? "Undefined"
          : `${formatNumber(metrics.angleDegrees, 1)}°`,
      label: "Angle between vectors",
    },
    {
      value: formatSignedNumber(metrics.dot, 3),
      label: "Dot product",
    },
    {
      value: formatNumber(metrics.magnitudeA, 3),
      label: "Magnitude of A",
    },
    {
      value: formatNumber(metrics.magnitudeB, 3),
      label: "Magnitude of B",
    },
  ];

  cards.forEach((cardInfo) => {
    const card = document.createElement("article");
    card.className = "cosine-metric-card";

    const value = document.createElement("strong");
    value.textContent = cardInfo.value;

    const label = document.createElement("span");
    label.textContent = cardInfo.label;

    card.append(value, label);
    cosineMetrics.appendChild(card);
  });
}

function renderCosineFormulaDisplay(metrics) {
  if (!cosineFormula) {
    return;
  }

  cosineFormula.replaceChildren();

  const lines = [
    `A = [${formatNumber(cosineVectors.a.x, 1)}, ${formatNumber(
      cosineVectors.a.y,
      1
    )}, ${formatNumber(cosineVectors.a.z, 1)}]`,
    `B = [${formatNumber(cosineVectors.b.x, 1)}, ${formatNumber(
      cosineVectors.b.y,
      1
    )}, ${formatNumber(cosineVectors.b.z, 1)}]`,
    `A·B = (${formatNumber(cosineVectors.a.x, 1)} * ${formatNumber(
      cosineVectors.b.x,
      1
    )}) + (${formatNumber(cosineVectors.a.y, 1)} * ${formatNumber(
      cosineVectors.b.y,
      1
    )}) + (${formatNumber(cosineVectors.a.z, 1)} * ${formatNumber(
      cosineVectors.b.z,
      1
    )}) = ${formatSignedNumber(metrics.dot, 3)}`,
    `|A| = sqrt(${formatNumber(cosineVectors.a.x * cosineVectors.a.x, 2)} + ${formatNumber(
      cosineVectors.a.y * cosineVectors.a.y,
      2
    )} + ${formatNumber(cosineVectors.a.z * cosineVectors.a.z, 2)}) = ${formatNumber(
      metrics.magnitudeA,
      3
    )}`,
    `|B| = sqrt(${formatNumber(cosineVectors.b.x * cosineVectors.b.x, 2)} + ${formatNumber(
      cosineVectors.b.y * cosineVectors.b.y,
      2
    )} + ${formatNumber(cosineVectors.b.z * cosineVectors.b.z, 2)}) = ${formatNumber(
      metrics.magnitudeB,
      3
    )}`,
    metrics.cosine === null
      ? "cos(theta) is undefined because one vector has zero length."
      : `cos(theta) = ${formatSignedNumber(metrics.dot, 3)} / (${formatNumber(
          metrics.magnitudeA,
          3
        )} * ${formatNumber(metrics.magnitudeB, 3)}) = ${formatNumber(
          metrics.cosine,
          3
        )}`,
  ];

  lines.forEach((text) => {
    const line = document.createElement("p");
    line.className = "formula-line";
    line.textContent = text;
    cosineFormula.appendChild(line);
  });
}

function renderCosineLab() {
  const metrics = {
    dot: dotProduct(cosineVectors.a, cosineVectors.b),
    magnitudeA: magnitude(cosineVectors.a),
    magnitudeB: magnitude(cosineVectors.b),
  };
  metrics.cosine = cosineSimilarity(cosineVectors.a, cosineVectors.b);
  metrics.angleDegrees =
    metrics.cosine === null
      ? null
      : (Math.acos(clamp(metrics.cosine, -1, 1)) * 180) / Math.PI;

  renderCosineMetricsDisplay(metrics);
  renderCosineFormulaDisplay(metrics);

  if (cosineDescription) {
    cosineDescription.textContent = describeCosineRelationship(metrics.cosine);
  }

  if (!cosineCanvas || !ensureCanvasSize(cosineCanvas)) {
    return;
  }

  const ctx = cosineCanvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  const width = cosineCanvas.width;
  const height = cosineCanvas.height;
  const scale = Math.min(width, height) / 7.5;
  const extent = COSINE_LIMIT + 0.2;
  const origin = project3DPoint({ x: 0, y: 0, z: 0 }, width, height, scale);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#eaf1ff";
  ctx.fillRect(0, 0, width, height);

  const axisColor = "rgba(13, 26, 53, 0.35)";
  const gridColor = "rgba(13, 26, 53, 0.12)";

  for (let tick = -COSINE_LIMIT; tick <= COSINE_LIMIT; tick += 1) {
    const xStart = project3DPoint({ x: tick, y: -extent, z: 0 }, width, height, scale);
    const xEnd = project3DPoint({ x: tick, y: extent, z: 0 }, width, height, scale);
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xStart.x, xStart.y);
    ctx.lineTo(xEnd.x, xEnd.y);
    ctx.stroke();
  }

  const axes = [
    { label: "X", to: { x: extent, y: 0, z: 0 }, color: axisColor },
    { label: "Y", to: { x: 0, y: extent, z: 0 }, color: axisColor },
    { label: "Z", to: { x: 0, y: 0, z: extent }, color: axisColor },
  ];

  axes.forEach((axis) => {
    const positive = project3DPoint(axis.to, width, height, scale);
    const negative = project3DPoint(
      { x: -axis.to.x, y: -axis.to.y, z: -axis.to.z },
      width,
      height,
      scale
    );

    ctx.strokeStyle = axis.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(negative.x, negative.y);
    ctx.lineTo(positive.x, positive.y);
    ctx.stroke();

    ctx.fillStyle = "#233456";
    ctx.font = `${13 * dpr}px "Segoe UI", Tahoma, Geneva, Verdana, sans-serif`;
    ctx.fillText(axis.label, positive.x + 8 * dpr, positive.y - 8 * dpr);
  });

  ctx.fillStyle = "#111c33";
  ctx.beginPath();
  ctx.arc(origin.x, origin.y, 5 * dpr, 0, Math.PI * 2);
  ctx.fill();

  const pointA = project3DPoint(cosineVectors.a, width, height, scale);
  const pointB = project3DPoint(cosineVectors.b, width, height, scale);

  drawArrow(ctx, origin, pointA, "#2563eb", 4 * dpr, "A", dpr);
  drawArrow(ctx, origin, pointB, "#f97316", 4 * dpr, "B", dpr);

  ctx.strokeStyle = "rgba(15, 23, 42, 0.14)";
  ctx.lineWidth = 2 * dpr;
  ctx.beginPath();
  ctx.moveTo(pointA.x, pointA.y);
  ctx.lineTo(pointB.x, pointB.y);
  ctx.stroke();
}

function syncCosineInputs(vectorKey, axisKey) {
  const refs = cosineControlRefs[vectorKey]?.[axisKey];
  if (!refs) {
    return;
  }

  const value = Number(cosineVectors[vectorKey][axisKey].toFixed(1));
  refs.range.value = String(value);
  refs.number.value = String(value);
}

function setCosineVectorValue(vectorKey, axisKey, nextValue) {
  const parsed = Number(nextValue);
  if (!Number.isFinite(parsed)) {
    return;
  }

  cosineVectors[vectorKey][axisKey] = clamp(parsed, -COSINE_LIMIT, COSINE_LIMIT);
  syncCosineInputs(vectorKey, axisKey);
  renderCosineLab();
}

function buildCosineControls(vectorKey, container) {
  if (!container) {
    return;
  }

  container.replaceChildren();

  COSINE_AXES.forEach((axis) => {
    const row = document.createElement("label");
    row.className = "axis-control";

    const axisLabel = document.createElement("span");
    axisLabel.className = "axis-label";
    axisLabel.textContent = axis.label;

    const range = document.createElement("input");
    range.type = "range";
    range.min = String(-COSINE_LIMIT);
    range.max = String(COSINE_LIMIT);
    range.step = "0.1";
    range.value = String(cosineVectors[vectorKey][axis.key]);

    const number = document.createElement("input");
    number.type = "number";
    number.min = String(-COSINE_LIMIT);
    number.max = String(COSINE_LIMIT);
    number.step = "0.1";
    number.value = String(cosineVectors[vectorKey][axis.key]);

    range.addEventListener("input", (event) => {
      setCosineVectorValue(vectorKey, axis.key, event.target.value);
    });
    number.addEventListener("input", (event) => {
      setCosineVectorValue(vectorKey, axis.key, event.target.value);
    });

    cosineControlRefs[vectorKey][axis.key] = { range, number };

    row.append(axisLabel, range, number);
    container.appendChild(row);
  });
}

function applyCosinePreset(presetName) {
  const preset = COSINE_PRESETS[presetName];
  if (!preset) {
    return;
  }

  cosineVectors = JSON.parse(JSON.stringify(preset));

  ["a", "b"].forEach((vectorKey) => {
    COSINE_AXES.forEach((axis) => {
      syncCosineInputs(vectorKey, axis.key);
    });
  });

  renderCosineLab();
}

function activateTab(targetId) {
  viewTabs.forEach((tab) => {
    const isActive = tab.dataset.tabTarget === targetId;
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });

  tabPanels.forEach((panel) => {
    const isActive = panel.id === targetId;
    panel.classList.toggle("is-active", isActive);
    panel.hidden = !isActive;
  });

  if (targetId === "embedding-map-panel") {
    renderMap(mapData);
  }

  if (targetId === "cosine-lab-panel") {
    renderCosineLab();
  }
}

async function runVisualization(queryText, queryVector = lastQueryVector) {
  const cleanedQuery = typeof queryText === "string" ? queryText.trim() : "";

  if (!cleanedQuery && !Array.isArray(queryVector)) {
    setMapStatus("Enter text to visualize embeddings.");
    return;
  }

  const limitValue = Number.parseInt(mapLimit?.value || "60", 10);
  const limit = Number.isNaN(limitValue) ? 60 : limitValue;
  const method = mapMethod?.value || "pca";
  const threshold = getSimilarityThreshold();

  setMapStatus(`Projecting vectors with ${method.toUpperCase()}...`);

  try {
    const payload = {
      queryText: cleanedQuery || lastQueryText,
      limit,
      method,
      within: threshold,
    };

    if (Array.isArray(queryVector)) {
      payload.vector = queryVector;
    }

    const response = await fetch(`${API_BASE_URL}/images/visualize/text`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const result = await parseJsonResponse(response);
    renderMap(result.points || []);
    const fallback = result.thresholdFallback
      ? " Backend retried without the threshold because no close matches were found."
      : "";
    setMapStatus(
      `${result.total || result.points?.length || 0} vectors projected via ${
        result.method?.toUpperCase() || method.toUpperCase()
      } with distance threshold ${formatThreshold(result.within ?? threshold)}.${fallback}`
    );
  } catch (error) {
    console.error("Visualization error:", error);
    setMapStatus(error.message || "Unable to visualize embeddings.");
  }
}

async function runTextSearch(queryText) {
  const query = typeof queryText === "string" ? queryText.trim() : "";
  if (!query) {
    return;
  }

  clearResults();
  setStatus("Searching by text...");
  setProcessStatus("Inspecting tokenization and vector-generation diagnostics...");

  try {
    const threshold = getSimilarityThreshold();
    const response = await fetch(`${API_BASE_URL}/get-image-by-text`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ queryText: query, within: threshold }),
    });
    const result = await parseJsonResponse(response);
    renderResults(result.matches);
    setStatus(formatSearchSummary(result));
    storeProcessResponse(result);
    await runVisualization(query, result.queryEmbedding?.vector);
  } catch (error) {
    console.error("Error:", error);
    setStatus(error.message || "Error processing search");
    setProcessStatus(error.message || "Unable to inspect the current query.");
  }
}

if (queryButton) {
  queryButton.addEventListener("click", () => {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.onchange = async (event) => {
      const file = event.target.files[0];
      if (!file) {
        return;
      }

      const formData = new FormData();
      formData.append("image", file);
      clearResults();
      setStatus("Searching by image...");

      try {
        const threshold = getSimilarityThreshold();
        const searchParams = new URLSearchParams({
          within: String(threshold),
        });
        const response = await fetch(
          `${API_BASE_URL}/get-image-by-image?${searchParams.toString()}`,
          {
            method: "POST",
            body: formData,
          }
        );
        const result = await parseJsonResponse(response);
        renderResults(result.matches);
        setStatus(formatSearchSummary(result));
      } catch (error) {
        console.error("Error:", error);
        setStatus(error.message || "Error processing image");
      }
    };
    fileInput.click();
  });
}

if (uploadButton) {
  uploadButton.addEventListener("click", () => {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.multiple = true;

    fileInput.onchange = async (event) => {
      const files = Array.from(event.target.files || []);
      if (!files.length) {
        return;
      }

      const formData = new FormData();
      files.forEach((file) => formData.append("images", file, file.name));

      setUploadButtonState(true, files.length);
      setStatus(`Uploading ${files.length} image${files.length === 1 ? "" : "s"}...`);

      try {
        const response = await fetch(`${API_BASE_URL}/images/upload`, {
          method: "POST",
          body: formData,
        });
        const result = await response.json();
        const summary = `${result.uploaded || 0} uploaded, ${result.failed || 0} failed.`;

        if (!response.ok) {
          throw new Error(result.error || summary || "Upload failed.");
        }

        setStatus(summary);
      } catch (error) {
        console.error("Upload error:", error);
        setStatus(error.message || "Error uploading images");
      } finally {
        setUploadButtonState(false);
      }
    };

    fileInput.click();
  });
}

if (queryInput) {
  queryInput.addEventListener("keypress", async (event) => {
    if (event.key === "Enter") {
      await runTextSearch(queryInput.value);
    }
  });
}

if (processRefreshButton) {
  processRefreshButton.addEventListener("click", async () => {
    await inspectCurrentQuery(queryInput?.value || lastQueryText);
  });
}

if (mapCanvas) {
  mapCanvas.addEventListener("mousemove", handleMapHover);
  mapCanvas.addEventListener("mouseleave", hideTooltip);
}

viewTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    activateTab(tab.dataset.tabTarget);
  });
});

if (mapButton) {
  mapButton.addEventListener("click", async () => {
    await runVisualization(lastQueryText || queryInput?.value, lastQueryVector);
  });
}

if (mapMethod) {
  mapMethod.addEventListener("change", async () => {
    if (lastQueryText || queryInput?.value) {
      await runVisualization(lastQueryText || queryInput.value, lastQueryVector);
    }
  });
}

if (thresholdSlider) {
  updateThresholdValue();
  thresholdSlider.addEventListener("input", updateThresholdValue);
}

cosinePresetButtons.forEach((button) => {
  button.addEventListener("click", () => {
    applyCosinePreset(button.dataset.cosinePreset);
  });
});

buildCosineControls("a", cosineControlsA);
buildCosineControls("b", cosineControlsB);
renderProcessInspector(null);
renderCosineLab();

window.addEventListener("resize", () => {
  renderMap(mapData);
  renderCosineLab();
});

if (appInfo && statusEl) {
  setStatus(`${appInfo.appName} v${appInfo.version} is ready.`);
} else if (statusEl) {
  setStatus("App bridge not detected.");
}
