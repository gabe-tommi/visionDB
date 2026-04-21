require("dotenv").config({ path: `${__dirname}/.env` });

const express = require("express");
const { getFirebaseAuth } = require("./firebaseAdmin");
const {
  resolveImageSearchEmbedding,
  resolveStoredEmbedding,
  resolveTextSearchEmbedding,
} = require("./embeddingService");
const {
  createImageRecord,
  deleteImagesByIds,
  getImageById,
  listImages,
  searchImagesByVector,
  updateImageRecord,
  upsertEmbeddingForImage,
} = require("./visionDbRepository");

const app = express();
const PORT = process.env.PORT || 3001;

app.use(express.json());

/*
 * Express entry point.
 *
 * This file is intentionally focused on HTTP concerns:
 * - validate/normalize request input
 * - verify auth tokens when present
 * - call the repository and embedding layers
 * - translate thrown errors into JSON responses
 * - load environment variables from `be/.env` before anything else runs
 *
 * Database access and embedding-specific logic live in separate modules so the
 * request handlers stay readable.
 */

/*
 * Creates an Error object that also carries an HTTP status code so the global
 * Express error middleware can return the right response.
 */
function createHttpError(statusCode, message) {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

/*
 * Tiny helper so async route handlers can throw normally and still end up in
 * Express error middleware.
 */
function asyncHandler(handler) {
  return async (req, res, next) => {
    try {
      await handler(req, res);
    } catch (error) {
      next(error);
    }
  };
}

/*
 * Lets query-string and JSON-body inputs share the same route handlers.
 *
 * Example:
 * - `tags=["cat","dog"]` from query params becomes an actual array
 * - `{ "ids": ["a", "b"] }` from JSON body passes through unchanged
 */
function parseJsonValue(value) {
  if (typeof value !== "string") {
    return value;
  }

  const trimmed = value.trim();

  if (!trimmed) {
    return value;
  }

  if (
    (trimmed.startsWith("[") && trimmed.endsWith("]")) ||
    (trimmed.startsWith("{") && trimmed.endsWith("}"))
  ) {
    try {
      return JSON.parse(trimmed);
    } catch (_error) {
      return value;
    }
  }

  return value;
}

/*
 * Merges query params and body into one payload object. Body values win if both
 * were supplied.
 */
function getPayload(req) {
  const source = {
    ...req.query,
    ...(req.body || {}),
  };

  return Object.fromEntries(
    Object.entries(source).map(([key, value]) => [key, parseJsonValue(value)])
  );
}

/*
 * Normalizers below convert flexible request formats into the shapes expected
 * by SQL Connect and the repository layer.
 */
function normalizeTags(tags) {
  if (tags === undefined) {
    return undefined;
  }

  if (Array.isArray(tags)) {
    return tags;
  }

  if (typeof tags === "string") {
    return tags
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);
  }

  throw createHttpError(400, "`tags` must be an array or comma-separated string.");
}

function normalizeMetadata(metadata) {
  if (metadata === undefined || metadata === null) {
    return metadata;
  }

  if (typeof metadata === "string") {
    return metadata;
  }

  return JSON.stringify(metadata);
}

function normalizeLimit(value, fallback) {
  if (value === undefined) {
    return fallback;
  }

  const parsed = Number(value);

  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw createHttpError(400, "`limit` must be a positive integer.");
  }

  return parsed;
}

function normalizeWithin(value) {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }

  const parsed = Number(value);

  if (Number.isNaN(parsed)) {
    throw createHttpError(400, "`within` must be a number.");
  }

  return parsed;
}

function normalizeIds(value, fallbackId) {
  if (Array.isArray(value)) {
    return value;
  }

  if (typeof value === "string" && value.trim()) {
    return value.includes(",")
      ? value
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean)
      : [value.trim()];
  }

  if (fallbackId) {
    return [fallbackId];
  }

  return [];
}

/*
 * Shapes the insert payload for the `Image` table.
 */
function buildImageRecord(payload) {
  return {
    imageUrl: payload.imageUrl,
    description: payload.description,
    tags: normalizeTags(payload.tags),
    metadata: normalizeMetadata(payload.metadata),
    createdAt: payload.createdAt || new Date().toISOString(),
  };
}

/*
 * Shapes the patch payload for image updates.
 */
function buildImagePatch(payload) {
  return {
    imageUrl: payload.imageUrl,
    description: payload.description,
    tags: normalizeTags(payload.tags),
    metadata: normalizeMetadata(payload.metadata),
    createdAt: payload.createdAt,
  };
}

/*
 * Optional Firebase Auth middleware.
 *
 * If a bearer token is present, we verify it and attach decoded claims to
 * `req.authClaims`. If `REQUIRE_FIREBASE_AUTH=true`, requests without a token
 * are rejected.
 *
 * Right now the decoded claims are available for future authorization logic,
 * even though the current repository calls run with admin privileges.
 */
async function attachAuthContext(req, _res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader) {
    if (process.env.REQUIRE_FIREBASE_AUTH === "true") {
      return next(
        createHttpError(401, "Missing Authorization header for protected route.")
      );
    }

    return next();
  }

  if (!authHeader.startsWith("Bearer ")) {
    return next(createHttpError(401, "Authorization header must use Bearer token."));
  }

  try {
    const idToken = authHeader.slice("Bearer ".length);
    req.authClaims = await getFirebaseAuth().verifyIdToken(idToken);
    return next();
  } catch (_error) {
    return next(createHttpError(401, "Invalid Firebase Authentication token."));
  }
}

app.use(attachAuthContext);

/*
 * Route handlers.
 *
 * These orchestrate the higher-level app flows:
 * - create image + embedding
 * - list images
 * - search by text
 * - search by image
 * - update image and optionally refresh embedding
 * - delete one or many images
 */
async function handleCreateImage(req, res) {
  const payload = getPayload(req);

  if (!payload.imageUrl) {
    throw createHttpError(400, "`imageUrl` is required.");
  }

  const imageRecord = buildImageRecord(payload);
  const embedding = await resolveStoredEmbedding(payload);

  const createdImage = await createImageRecord(imageRecord);

  await upsertEmbeddingForImage(createdImage.id, {
    vector: embedding.vector,
    dimension: embedding.dimension,
    modelUsed: embedding.modelUsed,
    generatedAt: payload.generatedAt || new Date().toISOString(),
  });

  const image = await getImageById(createdImage.id);

  res.status(201).json({ image });
}

async function handleListImages(req, res) {
  const payload = getPayload(req);
  const images = await listImages(normalizeLimit(payload.limit, 50));
  res.json({ images });
}

async function handleSearchByText(req, res) {
  const payload = getPayload(req);
  const embedding = await resolveTextSearchEmbedding(payload);
  const matches = await searchImagesByVector({
    vector: embedding.vector,
    limit: normalizeLimit(payload.limit, 5),
    within: normalizeWithin(payload.within),
  });

  res.json({ matches });
}

async function handleSearchByImage(req, res) {
  const payload = getPayload(req);
  const embedding = await resolveImageSearchEmbedding(payload);
  const matches = await searchImagesByVector({
    vector: embedding.vector,
    limit: normalizeLimit(payload.limit, 5),
    within: normalizeWithin(payload.within),
  });

  res.json({ matches });
}

async function handleUpdateImage(req, res) {
  const payload = getPayload(req);
  const imageId = req.params.id || payload.id;

  if (!imageId) {
    throw createHttpError(400, "`id` is required.");
  }

  const imagePatch = buildImagePatch(payload);
  await updateImageRecord(imageId, imagePatch);

  const wantsEmbeddingUpdate =
    payload.vector !== undefined ||
    payload.imageUrl !== undefined ||
    payload.description !== undefined;

  if (wantsEmbeddingUpdate) {
    const embedding = await resolveStoredEmbedding(payload);

    await upsertEmbeddingForImage(imageId, {
      vector: embedding.vector,
      dimension: embedding.dimension,
      modelUsed: embedding.modelUsed,
      generatedAt: payload.generatedAt || new Date().toISOString(),
    });
  }

  const image = await getImageById(imageId);

  if (!image) {
    throw createHttpError(404, `Image ${imageId} was not found.`);
  }

  res.json({ image });
}

async function handleDeleteImages(req, res) {
  const payload = getPayload(req);
  const ids = normalizeIds(payload.ids, req.params.id || payload.id);

  if (!ids.length) {
    throw createHttpError(400, "Provide `id` or `ids` to delete images.");
  }

  const deletedIds = await deleteImagesByIds(ids);
  res.json({ deletedIds });
}

app.get("/", (_req, res) => {
  res.json({ message: "VecDb backend is running" });
});

/*
 * Legacy routes keep your original API shape working.
 * New REST-style aliases are added below so the backend can evolve without
 * breaking your existing frontend calls.
 */
app.post("/add-image", asyncHandler(handleCreateImage));
app.get("/get-all-images", asyncHandler(handleListImages));
app.all("/get-image-by-image", asyncHandler(handleSearchByImage));
app.all("/get-image-by-text", asyncHandler(handleSearchByText));
app.patch("/update-image", asyncHandler(handleUpdateImage));
app.delete("/delete-image", asyncHandler(handleDeleteImages));

app.post("/images", asyncHandler(handleCreateImage));
app.get("/images", asyncHandler(handleListImages));
app.post("/images/search/image", asyncHandler(handleSearchByImage));
app.post("/images/search/text", asyncHandler(handleSearchByText));
app.patch("/images/:id", asyncHandler(handleUpdateImage));
app.delete("/images/:id", asyncHandler(handleDeleteImages));
app.delete("/images", asyncHandler(handleDeleteImages));

app.get("/health", (_req, res) => {
  res.status(200).json({ status: "ok" });
});

/*
 * Last-resort JSON error handler.
 *
 * Any thrown error from validation, auth, SQL Connect, or embedding generation
 * lands here and becomes a consistent `{ error: "..." }` payload.
 */
app.use((error, _req, res, _next) => {
  const statusCode = error.statusCode || 500;
  const message = error.message || "Internal server error";

  res.status(statusCode).json({ error: message });
});

app.listen(PORT, () => {
  console.log(`Backend listening on http://localhost:${PORT}`);
});
