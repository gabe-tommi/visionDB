const statusEl = document.getElementById("status");
const appInfo = window.vecdbApp;

if (appInfo && statusEl) {
  statusEl.textContent = `${appInfo.appName} v${appInfo.version} is ready.`;
} else if (statusEl) {
  statusEl.textContent = "App bridge not detected.";
}
