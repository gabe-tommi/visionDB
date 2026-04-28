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
 * Lazy-loaded, cached reference to the jina-clip-v2 feature-extraction pipeline.
 *
 * The model is downloaded from Hugging Face on the very first call
 * and cached in HF_HOME (default: ~/.cache/huggingface). Subsequent calls
 * reuse the same in-process pipeline instance.
 */
let _pipelinePromise = null;

async function getPipeline() {
  if (!_pipelinePromise) {
    // Dynamic import keeps the heavy library out of the module parse path
    // and lets the rest of the server start up before the model is needed.
    const { pipeline, env } = await import("@huggingface/transformers");

    // Resolve HF_HOME relative to this file's directory, stripping any
    // leading slash so it's always treated as a relative path.
    if (process.env.HF_HOME) {
      const relative = process.env.HF_HOME.replace(/^\/+/, "");
      env.cacheDir = require("path").resolve(__dirname, relative);
    }

    _pipelinePromise = pipeline("feature-extraction", "jinaai/jina-clip-v2", {
      dtype: "fp32",
    });
  }

  return _pipelinePromise;
}

/*
 * Extracts a flat JS number array from a Transformers.js tensor and truncates
 * it to EMBEDDING_DIMENSION using Matryoshka Representation Learning.
 *
 * jina-embeddings-v4 natively supports Matryoshka truncation: the first N
 * dimensions of the full 2048-dim output form a valid, independently useful
 * embedding. Slicing to 1024 retains strong retrieval quality while matching
 * the existing schema column size.
 */
function tensorToArray(tensor) {
  let full;

  // Mean-pool over the token dimension when shape is [1, tokens, dim].
  if (tensor.dims.length === 3) {
    const [, seqLen, dim] = tensor.dims;
    const data = tensor.data;
    const result = new Array(dim).fill(0);
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < dim; d++) {
        result[d] += data[t * dim + d];
      }
    }
    for (let d = 0; d < dim; d++) {
      result[d] /= seqLen;
    }
    full = result;
  } else {
    // Shape is already [1, dim] — squeeze batch dimension.
    full = Array.from(tensor.data);
  }

  // Matryoshka truncation: keep only the first EMBEDDING_DIMENSION values.
  return full.length > EMBEDDING_DIMENSION
    ? full.slice(0, EMBEDDING_DIMENSION)
    : full;
}

/*
 * Generates a 1024-dimensional text embedding using jina-clip-v2.
 *
 * The model outputs 1024 dims natively, matching the schema column size.
 */
async function generateTextEmbedding(text) {
  const extractor = await getPipeline();
  const output = await extractor(text, {
    pooling: "mean",
    normalize: true,
  });
  return tensorToArray(output);
}

/*
 * Generates a 1024-dimensional multimodal embedding using jina-clip-v2.
 *
 * Supports both text and image inputs. When an imageUrl is provided the model
 * encodes the image directly. Falls back to a text embedding of description
 * if image encoding fails.
 */
async function generateImageEmbedding({ imageUrl, description }) {
  const extractor = await getPipeline();
  const { RawImage } = await import("@huggingface/transformers");

  try {
    const image = await RawImage.fromURL(imageUrl);
    const output = await extractor(image, {
      pooling: "mean",
      normalize: true,
    });
    return tensorToArray(output);
  } catch (visionError) {
    // Vision encoder not available in this ONNX export — fall back to text.
    if (!description) {
      throw new Error(
        `Image embedding failed and no description was provided as fallback. Original error: ${visionError.message}`
      );
    }

    console.warn(
      "[embeddingService] Vision encoder unavailable, falling back to text embedding of description:",
      visionError.message
    );

    const output = await extractor(description, {
      pooling: "mean",
      normalize: true,
    });
    return tensorToArray(output);
  }
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
  getPipeline,
  resolveImageSearchEmbedding,
  resolveStoredEmbedding,
  resolveTextSearchEmbedding,
};
