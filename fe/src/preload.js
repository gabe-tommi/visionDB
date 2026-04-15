const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("vecdbApp", {
  appName: "VecDb Desktop",
  version: "0.1.0"
});
