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
 * Used for text embeddings only. The model is downloaded from Hugging Face
 * on the very first call and cached. Subsequent calls reuse the instance.
 */
let _pipelinePromise = null;

async function getPipeline() {
  if (!_pipelinePromise) {
    const { pipeline, env } = await import("@huggingface/transformers");

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
 * Lazy-loaded AutoModel + AutoProcessor for jina-clip-v2 image encoding.
 *
 * The pipeline API does not support RawImage inputs for this model.
 * Using AutoModel directly gives us the Jina image embedding outputs, already
 * pooled to [batch, 1024].
 */
let _visionModelPromise = null;

async function getVisionModel() {
  if (!_visionModelPromise) {
    const { AutoModel, AutoProcessor, env } = await import("@huggingface/transformers");

    if (process.env.HF_HOME) {
      const relative = process.env.HF_HOME.replace(/^\/+/, "");
      env.cacheDir = require("path").resolve(__dirname, relative);
    }

    const modelId = "jinaai/jina-clip-v2";
    _visionModelPromise = Promise.all([
      AutoModel.from_pretrained(modelId, { dtype: "fp32" }),
      AutoProcessor.from_pretrained(modelId),
    ]).then(([model, processor]) => ({ model, processor }));
  }

  return _visionModelPromise;
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
 * Generates a 1024-dimensional image embedding using jina-clip-v2.
 *
 * Uses AutoModel + AutoProcessor directly because the feature-extraction
 * pipeline does not support RawImage inputs. The model returns normalized
 * image embeddings pooled to shape [batch, 1024].
 */
async function generateImageEmbedding({ imageUrl, description }) {
  const { RawImage } = await import("@huggingface/transformers");

  try {
    console.log(`[embeddingService] Loading image from: ${imageUrl}`);
    const image = await RawImage.read(imageUrl);
    console.log(`[embeddingService] Image loaded (${image.width}x${image.height}), encoding...`);

    const { model, processor } = await getVisionModel();
    const inputs = await processor(null, [image], { padding: true, truncation: true });
    const output = await model(inputs);
    const imageEmbedding =
      output.l2norm_image_embeddings ||
      output.image_embeddings ||
      output.image_embeds;

    if (!imageEmbedding || !imageEmbedding.dims) {
      throw new Error(
        `image embedding missing from model output. Keys: ${Object.keys(output || {}).join(", ")}`
      );
    }

    // Image embedding shape is [batch=1, dim] — squeeze batch dimension.
    const full = Array.from(imageEmbedding.data);
    const vector = full.length > EMBEDDING_DIMENSION ? full.slice(0, EMBEDDING_DIMENSION) : full;
    console.log(`[embeddingService] Embedding OK — dim: ${vector.length}, first 5: [${vector.slice(0, 5).map(v => v.toFixed(6)).join(', ')}]`);
    return vector;
  } catch (visionError) {
    console.error(`[embeddingService] Image embedding threw:`, visionError);
    if (!description) {
      throw new Error(
        `Image embedding failed and no description was provided as fallback. Original error: ${visionError.message}`
      );
    }

    console.warn(
      "[embeddingService] Vision encoder failed, falling back to text embedding of description:",
      visionError.message
    );

    const extractor = await getPipeline();
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
