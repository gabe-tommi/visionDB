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

    details.append(title, score);
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
