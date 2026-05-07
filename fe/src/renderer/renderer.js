const statusEl = document.getElementById("status");
const appInfo = window.vecdbApp;
const API_BASE_URL = "http://localhost:3001";

const queryInput = document.getElementById("query-input");
const queryButton = document.getElementById("query-button");
const uploadButton = document.getElementById("upload-button");
const resultsEl = document.getElementById("results");
const mapMethod = document.getElementById("map-method");
const mapLimit = document.getElementById("map-limit");
const mapButton = document.getElementById("map-button");
const mapStatus = document.getElementById("map-status");
const mapCanvas = document.getElementById("map-canvas");
const mapTooltip = document.getElementById("map-tooltip");

let lastQueryText = "";
let mapPoints = [];
let mapData = [];

function setStatus(message) {
  if (statusEl) {
    statusEl.textContent = message;
  }
}

function setUploadButtonState(isUploading, total = 0) {
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

function getMatchImage(match) {
  return match.image || match;
}

function formatSimilarity(value) {
  if (typeof value !== "number") {
    return "";
  }

  return `${Math.round(value * 100)}% match`;
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

function ensureCanvasSize() {
  if (!mapCanvas) {
    return;
  }

  const rect = mapCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  mapCanvas.width = Math.max(1, Math.floor(rect.width * dpr));
  mapCanvas.height = Math.max(1, Math.floor(rect.height * dpr));
}

function colorForSimilarity(similarity) {
  if (typeof similarity !== "number") {
    return "#5c7cfa";
  }

  const hue = 210 - 210 * similarity;
  return `hsl(${hue.toFixed(0)}, 70%, 50%)`;
}

function renderMap(points) {
  if (!mapCanvas) {
    return;
  }

  ensureCanvasSize();

  const ctx = mapCanvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const width = mapCanvas.width;
  const height = mapCanvas.height;
  ctx.clearRect(0, 0, width, height);

  if (!points.length) {
    mapPoints = [];
    mapData = [];
    return;
  }

  mapData = points;

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
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

  const axisX =
    minX <= 0 && maxX >= 0 ? scaleX(0) : padding;
  const axisY =
    minY <= 0 && maxY >= 0 ? scaleY(0) : height - padding;

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
  for (let i = 0; i <= tickCount; i += 1) {
    const t = i / tickCount;
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
  const plotted = points.map((point) => ({
    ...point,
    screenX: scaleX(point.x) / dpr,
    screenY: scaleY(point.y) / dpr,
  }));

  plotted.forEach((point) => {
    const radius = point.kind === "query" ? 8 : 5;
    ctx.fillStyle =
      point.kind === "query" ? "#111c33" : colorForSimilarity(point.similarity);
    ctx.beginPath();
    ctx.arc(point.screenX * dpr, point.screenY * dpr, radius, 0, Math.PI * 2);
    ctx.fill();

    if (point.kind === "query") {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(point.screenX * dpr, point.screenY * dpr, radius + 2, 0, Math.PI * 2);
      ctx.stroke();
    }
  });

  mapPoints = plotted;
}

function showTooltip(point, event) {
  if (!mapTooltip || !point) {
    return;
  }

  const label = point.kind === "query" ? "Query" : point.label || "Result";
  const similarity =
    typeof point.similarity === "number"
      ? `${Math.round(point.similarity * 100)}% match`
      : "Similarity unavailable";

  mapTooltip.textContent = `${label} • ${similarity}`;
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
    return Math.sqrt(dx * dx + dy * dy) < 10;
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

queryButton.addEventListener("click", () => {
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append("image", file);
    clearResults();
    setStatus("Searching by image...");
    
    try {
      const response = await fetch(`${API_BASE_URL}/get-image-by-image`, {
        method: "POST",
        body: formData
      });
      const result = await parseJsonResponse(response);
      console.log("Result:", result);
      renderResults(result.matches);
      setStatus(`${result.total || result.matches?.length || 0} images ranked by similarity.`);
    } catch (error) {
      console.error("Error:", error);
      setStatus(error.message || "Error processing image");
    }
  };
  fileInput.click();
});

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

      console.log("Upload result:", result);
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

queryInput.addEventListener("keypress", async (event) => {
  if (event.key === "Enter") {
    const query = queryInput.value;
    if (!query) return;

    clearResults();
    setStatus("Searching by text...");

    try {
      const response = await fetch(`${API_BASE_URL}/get-image-by-text`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ queryText: query })
      });
      const result = await parseJsonResponse(response);
      console.log("Result:", result);
      renderResults(result.matches);
      setStatus(`${result.total || result.matches?.length || 0} images ranked by similarity.`);
      lastQueryText = query;
      await runVisualization(query);
    } catch (error) {
      console.error("Error:", error);
      setStatus(error.message || "Error processing search");
    }

  }
});

async function runVisualization(queryText) {
  if (!queryText) {
    setMapStatus("Enter text to visualize embeddings.");
    return;
  }

  const limitValue = Number.parseInt(mapLimit?.value || "60", 10);
  const limit = Number.isNaN(limitValue) ? 60 : limitValue;
  const method = mapMethod?.value || "pca";

  setMapStatus(`Projecting vectors with ${method.toUpperCase()}...`);

  try {
    const response = await fetch(`${API_BASE_URL}/images/visualize/text`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        queryText,
        limit,
        method,
      }),
    });

    const result = await parseJsonResponse(response);
    renderMap(result.points || []);
    setMapStatus(
      `${result.total || result.points?.length || 0} vectors projected via ${result.method?.toUpperCase() || method.toUpperCase()}.`
    );
  } catch (error) {
    console.error("Visualization error:", error);
    setMapStatus(error.message || "Unable to visualize embeddings.");
  }
}

if (mapCanvas) {
  mapCanvas.addEventListener("mousemove", handleMapHover);
  mapCanvas.addEventListener("mouseleave", hideTooltip);
  window.addEventListener("resize", () => renderMap(mapData));
}

if (mapButton) {
  mapButton.addEventListener("click", async () => {
    await runVisualization(lastQueryText || queryInput.value);
  });
}

if (mapMethod) {
  mapMethod.addEventListener("change", async () => {
    if (lastQueryText || queryInput.value) {
      await runVisualization(lastQueryText || queryInput.value);
    }
  });
}

if (appInfo && statusEl) {
  setStatus(`${appInfo.appName} v${appInfo.version} is ready.`);
} else if (statusEl) {
  setStatus("App bridge not detected.");
}
