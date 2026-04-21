const express = require("express");

const app = express();
const PORT = process.env.PORT || 3001;

app.use(express.json());

app.get("/", (_req, res) => {
  res.json({ message: "VecDb backend is running" });
});

app.post("/add-image", (_req, res) => {
  // TODO: post method, read user image, create embedding, and add to database
  res.json({ message: "Added Image to database" });
});

app.get("/get-all-images", (_req, res) => {
  // TODO: get method, returns all images
  res.json({ message: "Retrieved All Images" });
});

app.get("/get-image-by-image", (_req, res) => {
  // TODO: get method, read user image, create embedding, and search database for similar images
  res.json({ message: "Retrieved Image by Image" });
});

app.get("/get-image-by-text", (_req, res) => {
  // TODO: get method, read user text, create embedding, and search database for similar images
  res.json({ message: "Retrieved Image by Text" });
});

app.patch("/update-image", (_req, res) => {
  // TODO: patch method, check what data is going to be updated, if image is updated recreate embedding
  res.json({ message: "Updated Image in database" });
});

app.delete("/delete-image", (_req, res) => {
  // TODO: delete method, deletes image from database. 
  // User passes in array with at least one id, and it needs to parse through all IDs to get rid of all the images
  res.json({ message: "Deleted Image from database" });
});

app.get("/health", (_req, res) => {
  res.status(200).json({ status: "ok" });
});

app.listen(PORT, () => {
  console.log(`Backend listening on http://localhost:${PORT}`);
});
