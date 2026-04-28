const statusEl = document.getElementById("status");
const appInfo = window.vecdbApp;
const API_BASE_URL = "http://localhost:3001";

const queryInput = document.getElementById("query-input");
const queryButton = document.getElementById("query-button");
const uploadButton = document.getElementById("upload-button");

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

queryButton.addEventListener("click", () => {
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append("image", file);
    
    try {
      const response = await fetch(`${API_BASE_URL}/get-image-by-image`, {
        method: "POST",
        body: formData
      });
      const result = await response.json();
      console.log("Result:", result);
      setStatus(JSON.stringify(result));
    } catch (error) {
      console.error("Error:", error);
      setStatus("Error processing image");
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

    // Handle the query input here
    try {
      const response = await fetch(`${API_BASE_URL}/get-image-by-text`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ queryText: query })
      });
      const result = await response.json();
      console.log("Result:", result);
      setStatus(JSON.stringify(result));
    } catch (error) {
      console.error("Error:", error);
      setStatus("Error processing search");
    }

  }
});

if (appInfo && statusEl) {
  setStatus(`${appInfo.appName} v${appInfo.version} is ready.`);
} else if (statusEl) {
  setStatus("App bridge not detected.");
}
