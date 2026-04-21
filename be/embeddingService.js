const EMBEDDING_DIMENSION = 1024;

/*
 * Embedding boundary for the backend.
 *
 * This file exists so the rest of the app does not care whether embeddings
 * came from:
 * - the client sending a ready-made vector
 * - a text embedding model
 * - an image embedding model
 *
 * Right now the automatic generators are intentionally placeholders, so the
 * server can already store/search vectors while you decide what model service
 * you want to plug in.
 */

/*
 * Ensures vectors match your SQL Connect schema exactly.
 *
 * Your `Embedding.vector` column is declared as `Vector @col(size: 1024)`, so
 * anything shorter or longer would fail conceptually and likely at runtime.
 */
function normalizeVector(vector) {
  if (!Array.isArray(vector)) {
    throw new Error("Embedding vector must be an array of numbers.");
  }

  if (vector.length !== EMBEDDING_DIMENSION) {
    throw new Error(
      `Embedding vector must contain ${EMBEDDING_DIMENSION} values.`
    );
  }

  const normalized = vector.map((value) => Number(value));

  if (normalized.some((value) => Number.isNaN(value))) {
    throw new Error("Embedding vector contains a non-numeric value.");
  }

  return normalized;
}

/*
 * Tracks which embedding model produced a vector.
 *
 * This is useful because vectors from different model families or versions are
 * usually not comparable in a meaningful way, even if they have the same size.
 */
function ensureModelUsed(modelUsed) {
  const value = modelUsed || process.env.VISIONDB_EMBEDDING_MODEL;

  if (!value) {
    throw new Error(
      "Missing embedding model. Provide `modelUsed` in the request or set VISIONDB_EMBEDDING_MODEL."
    );
  }

  return value;
}

/*
 * Placeholder hook for text embedding generation.
 *
 * Replace this with your actual model call once you decide where embeddings are
 * produced. For now, the route can still work if the client sends `vector`.
 */
async function generateTextEmbedding(_text) {
  throw new Error(
    "Automatic text embedding generation is not configured yet. Pass a 1024-length `vector`, or implement generateTextEmbedding in /Users/ezedi/Documents/GitHub/visionDB/be/embeddingService.js."
  );
}

/*
 * Placeholder hook for image embedding generation.
 *
 * Replace this with your real image embedding pipeline. The rest of the backend
 * is already shaped so you only need to change this function later.
 */
async function generateImageEmbedding(_input) {
  throw new Error(
    "Automatic image embedding generation is not configured yet. Pass a 1024-length `vector`, or implement generateImageEmbedding in /Users/ezedi/Documents/GitHub/visionDB/be/embeddingService.js."
  );
}

/*
 * Normalizes whatever the embedding generator returned into one consistent
 * payload used by the rest of the backend.
 */
function finalizeEmbeddingResult(result, fallbackModelUsed) {
  if (Array.isArray(result)) {
    return {
      vector: normalizeVector(result),
      dimension: EMBEDDING_DIMENSION,
      modelUsed: ensureModelUsed(fallbackModelUsed),
    };
  }

  if (result && typeof result === "object" && Array.isArray(result.vector)) {
    return {
      vector: normalizeVector(result.vector),
      dimension: EMBEDDING_DIMENSION,
      modelUsed: ensureModelUsed(result.modelUsed || fallbackModelUsed),
    };
  }

  throw new Error("Embedding generator returned an invalid payload.");
}

/*
 * Used during create/update flows.
 *
 * Priority order:
 * 1. use a caller-provided vector
 * 2. generate from image input if available
 * 3. generate from description text if available
 */
async function resolveStoredEmbedding(payload) {
  if (payload.vector) {
    return finalizeEmbeddingResult(payload.vector, payload.modelUsed);
  }

  if (payload.imageUrl) {
    return finalizeEmbeddingResult(
      await generateImageEmbedding({
        imageUrl: payload.imageUrl,
        description: payload.description,
      }),
      payload.modelUsed
    );
  }

  if (payload.description) {
    return finalizeEmbeddingResult(
      await generateTextEmbedding(payload.description),
      payload.modelUsed
    );
  }

  throw new Error(
    "No embedding input provided. Pass `vector`, or implement automatic embedding generation."
  );
}

/*
 * Used by text similarity search routes.
 */
async function resolveTextSearchEmbedding(payload) {
  if (payload.vector) {
    return finalizeEmbeddingResult(payload.vector, payload.modelUsed);
  }

  if (!payload.queryText) {
    throw new Error("Missing `queryText` for text similarity search.");
  }

  return finalizeEmbeddingResult(
    await generateTextEmbedding(payload.queryText),
    payload.modelUsed
  );
}

/*
 * Used by image similarity search routes.
 */
async function resolveImageSearchEmbedding(payload) {
  if (payload.vector) {
    return finalizeEmbeddingResult(payload.vector, payload.modelUsed);
  }

  if (!payload.imageUrl) {
    throw new Error("Missing `imageUrl` for image similarity search.");
  }

  return finalizeEmbeddingResult(
    await generateImageEmbedding({
      imageUrl: payload.imageUrl,
      description: payload.description,
    }),
    payload.modelUsed
  );
}

module.exports = {
  EMBEDDING_DIMENSION,
  resolveImageSearchEmbedding,
  resolveStoredEmbedding,
  resolveTextSearchEmbedding,
};
