const statusEl = document.getElementById("status");
const appInfo = window.vecdbApp;

const queryInput = document.getElementById("query-input");
const queryButton = document.getElementById("query-button");

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
      const response = await fetch("http://localhost:3000/get-image-by-image", {
        method: "POST",
        body: formData
      });
      const result = await response.json();
      console.log("Result:", result);
      statusEl.textContent = JSON.stringify(result);
    } catch (error) {
      console.error("Error:", error);
      statusEl.textContent = "Error processing image";
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
      const response = await fetch("http://localhost:3000/get-image-by-text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: query })
      });
      const result = await response.json();
      console.log("Result:", result);
      statusEl.textContent = JSON.stringify(result);
    } catch (error) {
      console.error("Error:", error);
      statusEl.textContent = "Error processing search";
    }

  }
});

if (appInfo && statusEl) {
  statusEl.textContent = `${appInfo.appName} v${appInfo.version} is ready.`;
} else if (statusEl) {
  statusEl.textContent = "App bridge not detected.";
}

