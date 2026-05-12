require("dotenv").config({ path: `${__dirname}/.env` });


const express = require("express");
const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { getFirebaseAuth, getStorageBucket } = require("./firebaseAdmin");
const {
  getPipeline,
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
const {
  buildProjectionPoints,
  normalizeMethod,
  projectVectors,
  SUPPORTED_METHODS,
} = require("./vectorProjection");
const { seedSampleImages } = require("./seedImages");

const app = express();
const PORT = process.env.PORT || 3001;
const DEFAULT_VECTOR_DISTANCE_THRESHOLD = 0.35;

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader(
    "Access-Control-Allow-Methods",
    "GET, POST, PATCH, DELETE, OPTIONS"
  );

  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }

  next();
});

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

function normalizeWithin(value, fallback = DEFAULT_VECTOR_DISTANCE_THRESHOLD) {
  if (value === undefined || value === null || value === "") {
    return fallback;
  }

  const parsed = Number(value);

  if (!Number.isFinite(parsed) || parsed < 0) {
    throw createHttpError(400, "`within` must be a non-negative number.");
  }

  return parsed;
}

async function searchImagesByVectorWithFallback({ vector, limit, within }) {
  const matches = await searchImagesByVector({ vector, limit, within });

  if (matches.length || within === undefined) {
    console.log(
      `[search] threshold ${within === undefined ? "disabled" : `within=${within}`} returned ${matches.length} match(es).`
    );

    return {
      matches,
      thresholdApplied: within !== undefined,
      thresholdFallback: false,
    };
  }

  console.log(
    `[search] threshold within=${within} returned 0 matches; retrying without threshold.`
  );

  const fallbackMatches = await searchImagesByVector({
    vector,
    limit,
    within: undefined,
  });

  console.log(
    `[search] fallback without threshold returned ${fallbackMatches.length} match(es).`
  );

  return {
    matches: fallbackMatches,
    thresholdApplied: false,
    thresholdFallback: true,
  };
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

function safeStorageName(filename) {
  return path
    .basename(filename || "image")
    .replace(/[^a-z0-9._-]+/gi, "-")
    .replace(/^-+|-+$/g, "") || "image";
}

function descriptionFromFilename(filename) {
  return path
    .basename(filename, path.extname(filename))
    .replace(/[_-]+/g, " ")
    .trim();
}

function parseContentDisposition(value) {
  const result = {};

  for (const part of value.split(";")) {
    const [rawKey, ...rawValue] = part.trim().split("=");
    if (!rawValue.length) {
      continue;
    }

    result[rawKey.toLowerCase()] = rawValue
      .join("=")
      .trim()
      .replace(/^"|"$/g, "");
  }

  return result;
}

function splitMultipartBody(body, boundary) {
  const delimiter = Buffer.from(`--${boundary}`);
  const parts = [];
  let searchFrom = 0;

  while (searchFrom < body.length) {
    const start = body.indexOf(delimiter, searchFrom);
    if (start === -1) {
      break;
    }

    const nextStart = body.indexOf(delimiter, start + delimiter.length);
    if (nextStart === -1) {
      break;
    }

    let partStart = start + delimiter.length;
    if (body[partStart] === 45 && body[partStart + 1] === 45) {
      break;
    }

    if (body[partStart] === 13 && body[partStart + 1] === 10) {
      partStart += 2;
    }

    let partEnd = nextStart;
    if (body[partEnd - 2] === 13 && body[partEnd - 1] === 10) {
      partEnd -= 2;
    }

    parts.push(body.subarray(partStart, partEnd));
    searchFrom = nextStart;
  }

  return parts;
}

function parseMultipartFiles(req) {
  return new Promise((resolve, reject) => {
    const contentType = req.headers["content-type"] || "";
    const match = contentType.match(/boundary=(?:"([^"]+)"|([^;]+))/i);

    if (!match) {
      reject(createHttpError(400, "Expected multipart form data with a boundary."));
      return;
    }

    const boundary = match[1] || match[2];
    const chunks = [];

    req.on("data", (chunk) => chunks.push(chunk));
    req.on("error", reject);
    req.on("end", () => {
      const body = Buffer.concat(chunks);
      const files = [];

      for (const part of splitMultipartBody(body, boundary)) {
        const headerEnd = part.indexOf(Buffer.from("\r\n\r\n"));
        if (headerEnd === -1) {
          continue;
        }

        const rawHeaders = part.subarray(0, headerEnd).toString("utf8");
        const content = part.subarray(headerEnd + 4);
        const headers = Object.fromEntries(
          rawHeaders.split("\r\n").map((line) => {
            const separator = line.indexOf(":");
            return [
              line.slice(0, separator).trim().toLowerCase(),
              line.slice(separator + 1).trim(),
            ];
          })
        );
        const disposition = parseContentDisposition(
          headers["content-disposition"] || ""
        );

        if (!disposition.filename || !content.length) {
          continue;
        }

        files.push({
          fieldName: disposition.name,
          originalName: disposition.filename,
          contentType: headers["content-type"] || "application/octet-stream",
          buffer: content,
        });
      }

      resolve(files);
    });
  });
}

function isMultipartRequest(req) {
  return (req.headers["content-type"] || "").startsWith("multipart/form-data");
}

async function uploadBufferToFirebase({ buffer, filename, contentType }) {
  const bucket = getStorageBucket();
  const safeName = safeStorageName(filename);
  const destination = `uploadedImages/${Date.now()}-${crypto.randomUUID()}-${safeName}`;
  const file = bucket.file(destination);

  await file.save(buffer, {
    metadata: { contentType },
    resumable: false,
  });
  await file.makePublic();

  return {
    file,
    imageUrl: `https://storage.googleapis.com/${bucket.name}/${destination}`,
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
  const within = normalizeWithin(payload.within);
  const searchResult = await searchImagesByVectorWithFallback({
    vector: embedding.vector,
    limit: normalizeLimit(payload.limit, 50),
    within,
  });

  res.json({
    matches: searchResult.matches,
    total: searchResult.matches.length,
    within,
    thresholdApplied: searchResult.thresholdApplied,
    thresholdFallback: searchResult.thresholdFallback,
  });
}

async function handleVisualizeByText(req, res) {
  const payload = getPayload(req);

  if (!payload.queryText) {
    throw createHttpError(400, "`queryText` is required.");
  }

  const method = normalizeMethod(payload.method);
  const embedding = await resolveTextSearchEmbedding(payload);
  const within = normalizeWithin(payload.within);
  const searchResult = await searchImagesByVectorWithFallback({
    vector: embedding.vector,
    limit: normalizeLimit(payload.limit, 60),
    within,
  });

  const items = buildProjectionPoints({
    queryVector: embedding.vector,
    queryLabel: payload.queryText,
    matches: searchResult.matches,
  });

  const vectors = items.map((item) => item.vector);
  const coords = await projectVectors(vectors, method);

  const points = items.map((item, index) => ({
    ...item,
    x: coords[index]?.[0] ?? 0,
    y: coords[index]?.[1] ?? 0,
  }));

  res.json({
    method,
    methods: Array.from(SUPPORTED_METHODS),
    total: points.length,
    within,
    thresholdApplied: searchResult.thresholdApplied,
    thresholdFallback: searchResult.thresholdFallback,
    points,
  });
}

async function handleSearchByImage(req, res) {
  if (isMultipartRequest(req)) {
    const files = await parseMultipartFiles(req);
    const file = files[0];

    if (!file) {
      throw createHttpError(400, "Select an image file to search with.");
    }

    const tempDir = await fs.promises.mkdtemp(
      path.join(os.tmpdir(), "visiondb-search-")
    );
    const tempPath = path.join(tempDir, safeStorageName(file.originalName));

    try {
      await fs.promises.writeFile(tempPath, file.buffer);

      const embedding = await resolveImageSearchEmbedding({
        imageUrl: tempPath,
        description: descriptionFromFilename(file.originalName),
      });
      const within = normalizeWithin(req.query.within);
      const searchResult = await searchImagesByVectorWithFallback({
        vector: embedding.vector,
        limit: normalizeLimit(req.query.limit, 50),
        within,
      });

      res.json({
        matches: searchResult.matches,
        total: searchResult.matches.length,
        within,
        thresholdApplied: searchResult.thresholdApplied,
        thresholdFallback: searchResult.thresholdFallback,
      });
      return;
    } finally {
      await fs.promises.rm(tempDir, { recursive: true, force: true });
    }
  }

  const payload = getPayload(req);
  const embedding = await resolveImageSearchEmbedding(payload);
  const within = normalizeWithin(payload.within);
  const searchResult = await searchImagesByVectorWithFallback({
    vector: embedding.vector,
    limit: normalizeLimit(payload.limit, 50),
    within,
  });

  res.json({
    matches: searchResult.matches,
    total: searchResult.matches.length,
    within,
    thresholdApplied: searchResult.thresholdApplied,
    thresholdFallback: searchResult.thresholdFallback,
  });
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

async function handleUploadImages(req, res) {
  const files = await parseMultipartFiles(req);

  if (!files.length) {
    throw createHttpError(400, "Select at least one image file to upload.");
  }

  const results = [];

  for (const file of files) {
    const tempDir = await fs.promises.mkdtemp(
      path.join(os.tmpdir(), "visiondb-upload-")
    );
    const tempPath = path.join(tempDir, safeStorageName(file.originalName));
    let uploadedStorageFile = null;

    try {
      if (!file.contentType.startsWith("image/")) {
        throw createHttpError(400, `${file.originalName} is not an image.`);
      }

      await fs.promises.writeFile(tempPath, file.buffer);

      const description = descriptionFromFilename(file.originalName);
      const embedding = await resolveStoredEmbedding({
        imageUrl: tempPath,
        description,
      });
      const uploaded = await uploadBufferToFirebase({
        buffer: file.buffer,
        filename: file.originalName,
        contentType: file.contentType,
      });
      uploadedStorageFile = uploaded.file;
      const createdImage = await createImageRecord({
        imageUrl: uploaded.imageUrl,
        description,
        metadata: JSON.stringify({
          originalName: file.originalName,
          contentType: file.contentType,
          size: file.buffer.length,
        }),
        createdAt: new Date().toISOString(),
      });

      await upsertEmbeddingForImage(createdImage.id, {
        vector: embedding.vector,
        dimension: embedding.dimension,
        modelUsed: embedding.modelUsed,
        generatedAt: new Date().toISOString(),
      });

      results.push({
        filename: file.originalName,
        status: "uploaded",
        image: await getImageById(createdImage.id),
      });
    } catch (error) {
      if (uploadedStorageFile) {
        try {
          await uploadedStorageFile.delete({ ignoreNotFound: true });
        } catch (deleteError) {
          console.warn(
            `[upload] Could not clean up ${file.originalName}: ${deleteError.message}`
          );
        }
      }

      results.push({
        filename: file.originalName,
        status: "failed",
        error: error.message,
      });
    } finally {
      await fs.promises.rm(tempDir, { recursive: true, force: true });
    }
  }

  const uploaded = results.filter((result) => result.status === "uploaded");
  const failed = results.filter((result) => result.status === "failed");
  const statusCode = uploaded.length ? 201 : 400;

  res.status(statusCode).json({
    uploaded: uploaded.length,
    failed: failed.length,
    results,
  });
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
app.post("/upload-images", asyncHandler(handleUploadImages));
app.get("/get-all-images", asyncHandler(handleListImages));
app.all("/get-image-by-image", asyncHandler(handleSearchByImage));
app.all("/get-image-by-text", asyncHandler(handleSearchByText));
app.post("/visualize-text", asyncHandler(handleVisualizeByText));
app.patch("/update-image", asyncHandler(handleUpdateImage));
app.delete("/delete-image", asyncHandler(handleDeleteImages));

app.post("/images", asyncHandler(handleCreateImage));
app.post("/images/upload", asyncHandler(handleUploadImages));
app.get("/images", asyncHandler(handleListImages));
app.post("/images/search/image", asyncHandler(handleSearchByImage));
app.post("/images/search/text", asyncHandler(handleSearchByText));
app.post("/images/visualize/text", asyncHandler(handleVisualizeByText));
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

  console.error("[server]", error.stack || error);

  res.status(statusCode).json({ error: message });
});

app.listen(PORT, () => {
  console.log(`Backend listening on http://localhost:${PORT}`);
  if (process.env.HF_HOME) {
    // Strip any leading slash so the value is always treated as relative to
    // the server's directory, regardless of how it was written in .env.
    const relative = process.env.HF_HOME.replace(/^\/+/, "");
    const cacheDir = path.resolve(__dirname, relative);
    fs.mkdirSync(cacheDir, { recursive: true });
    console.log(`[embeddingService] Model cache directory: ${cacheDir}`);
  }
  // Trigger model download/load at startup so the first real request
  // doesn't stall while the ~2-3 GB Jina v4 weights are fetched.
  getPipeline()
    .then(() => {
      console.log("[embeddingService] jina-clip-v2 ready.");
      return seedSampleImages();
    })
    .catch((err) => console.error("[embeddingService] Model warmup failed:", err.message));

  });
