const statusEl = document.getElementById("status");
const appInfo = window.vecdbApp;
const API_BASE_URL = "http://localhost:3001";

const queryInput = document.getElementById("query-input");
const queryButton = document.getElementById("query-button");
const uploadButton = document.getElementById("upload-button");
const resultsEl = document.getElementById("results");

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
    } catch (error) {
      console.error("Error:", error);
      setStatus(error.message || "Error processing search");
    }

  }
});

if (appInfo && statusEl) {
  setStatus(`${appInfo.appName} v${appInfo.version} is ready.`);
} else if (statusEl) {
  setStatus("App bridge not detected.");
}
