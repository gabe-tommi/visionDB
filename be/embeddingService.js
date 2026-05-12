const EMBEDDING_DIMENSION = 1024;
const MODEL_ID = "jinaai/jina-clip-v2";

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
 * Lazy-loaded text encoder for jina-clip-v2.
 *
 * We use AutoTokenizer + AutoModel directly instead of the higher-level
 * pipeline helper because the pipeline's built-in mean pooling path has been
 * throwing on this model during text search in this app.
 */
let _textModelPromise = null;

async function getPipeline() {
  if (!_textModelPromise) {
    const { AutoModel, AutoTokenizer, env } = await import("@huggingface/transformers");

    if (process.env.HF_HOME) {
      const relative = process.env.HF_HOME.replace(/^\/+/, "");
      env.cacheDir = require("path").resolve(__dirname, relative);
    }

    _textModelPromise = Promise.all([
      AutoModel.from_pretrained(MODEL_ID, { dtype: "fp32" }),
      AutoTokenizer.from_pretrained(MODEL_ID),
    ]).then(([model, tokenizer]) => ({ model, tokenizer }));
  }

  return _textModelPromise;
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

    _visionModelPromise = Promise.all([
      AutoModel.from_pretrained(MODEL_ID, { dtype: "fp32" }),
      AutoProcessor.from_pretrained(MODEL_ID),
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

function normalizeL2(vector) {
  const magnitude = Math.sqrt(
    vector.reduce((sum, value) => sum + value * value, 0)
  );

  if (!magnitude) {
    return vector;
  }

  return vector.map((value) => value / magnitude);
}

function calculateMagnitude(vector) {
  return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
}

function summarizeVector(vector, previewSize = 48, topCount = 8) {
  const preview = vector.slice(0, previewSize);
  const magnitude = calculateMagnitude(vector);
  const min = Math.min(...vector);
  const max = Math.max(...vector);
  const mean = vector.reduce((sum, value) => sum + value, 0) / vector.length;
  const positiveCount = vector.filter((value) => value > 0).length;
  const negativeCount = vector.filter((value) => value < 0).length;

  const topDimensions = vector
    .map((value, index) => ({
      index,
      value,
      absoluteValue: Math.abs(value),
    }))
    .sort((a, b) => b.absoluteValue - a.absoluteValue)
    .slice(0, topCount)
    .map(({ index, value }) => ({ index, value }));

  return {
    preview,
    magnitude,
    min,
    max,
    mean,
    positiveCount,
    negativeCount,
    topDimensions,
  };
}

function tensorRowToArray(tensor) {
  if (!tensor?.tolist) {
    return null;
  }

  const value = tensor.tolist();
  return Array.isArray(value?.[0]) ? value[0].map((item) => Number(item)) : value;
}

function extractTextEmbeddingOutput(output) {
  const textEmbedding =
    output.l2norm_text_embeddings ||
    output.text_embeddings ||
    output.text_embeds;

  if (textEmbedding?.data) {
    const full = Array.from(textEmbedding.data);
    return {
      vector:
        full.length > EMBEDDING_DIMENSION
          ? full.slice(0, EMBEDDING_DIMENSION)
          : full,
      tensor: textEmbedding,
      source: output.l2norm_text_embeddings
        ? "l2norm_text_embeddings"
        : output.text_embeddings
          ? "text_embeddings"
          : "text_embeds",
    };
  }

  if (output.last_hidden_state?.data) {
    return {
      vector: normalizeL2(tensorToArray(output.last_hidden_state)),
      tensor: output.last_hidden_state,
      source: "last_hidden_state_mean_pool",
    };
  }

  throw new Error(
    `text embedding missing from model output. Keys: ${Object.keys(output || {}).join(", ")}`
  );
}

function buildTokenDiagnostics({
  tokenizer,
  inputIds,
  tokenStrings,
  attentionMask,
}) {
  const specialIds = new Set((tokenizer.all_special_ids || []).map((id) => Number(id)));

  return inputIds.map((id, index) => ({
    index,
    id: Number(id),
    text:
      tokenStrings[index] ||
      tokenizer.decode([id], {
        skip_special_tokens: false,
        clean_up_tokenization_spaces: false,
      }) ||
      `[${id}]`,
    isSpecial: specialIds.has(Number(id)),
    attended: Number(attentionMask[index] ?? 1) === 1,
  }));
}

async function runTextEmbedding(text, { includeDiagnostics = false } = {}) {
  const cleanedText = typeof text === "string" ? text.trim() : "";

  if (!cleanedText) {
    throw new Error("Text is required to generate an embedding.");
  }

  const { model, tokenizer } = await getPipeline();
  const rawTokenIds = tokenizer.encode(cleanedText, { add_special_tokens: true });
  const rawTokenStrings = tokenizer.tokenize(cleanedText, {
    add_special_tokens: true,
  });
  const modelInputs = await tokenizer([cleanedText], { padding: true, truncation: true });
  const output = await model(modelInputs);
  const extracted = extractTextEmbeddingOutput(output);

  if (!includeDiagnostics) {
    return extracted;
  }

  const inputIds = tensorRowToArray(modelInputs.input_ids) || rawTokenIds.map((id) => Number(id));
  const attentionMask =
    tensorRowToArray(modelInputs.attention_mask) ||
    inputIds.map(() => 1);
  const tokens = buildTokenDiagnostics({
    tokenizer,
    inputIds,
    tokenStrings: rawTokenStrings,
    attentionMask,
  });
  const embeddingStats = summarizeVector(extracted.vector);
  const truncatedTokenCount = Math.max(0, rawTokenIds.length - inputIds.length);

  return {
    ...extracted,
    diagnostics: {
      queryText: cleanedText,
      modelUsed: MODEL_ID,
      generatedAt: new Date().toISOString(),
      tokenization: {
        tokens,
        totalTokens: tokens.length,
        specialTokenCount: tokens.filter((token) => token.isSpecial).length,
        attendedTokenCount: attentionMask.filter((value) => Number(value) === 1).length,
        truncatedTokenCount,
        hasPadding: attentionMask.some((value) => Number(value) === 0),
        attentionMask,
      },
      tensors: {
        inputIdsShape: modelInputs.input_ids?.dims || [1, inputIds.length],
        attentionMaskShape: modelInputs.attention_mask?.dims || [1, attentionMask.length],
        embeddingShape: extracted.tensor?.dims || [1, extracted.vector.length],
        hiddenStateShape: output.last_hidden_state?.dims || null,
        embeddingSource: extracted.source,
      },
      embedding: {
        dimension: extracted.vector.length,
        ...embeddingStats,
      },
    },
  };
}

/*
 * Generates a 1024-dimensional text embedding using jina-clip-v2.
 *
 * The model outputs 1024 dims natively, matching the schema column size.
 */
async function generateTextEmbedding(text) {
  const result = await runTextEmbedding(text);
  return result.vector;
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

    return generateTextEmbedding(description);
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

async function inspectTextEmbedding(payload) {
  if (payload.vector) {
    const finalized = finalizeEmbeddingResult(payload.vector, payload.modelUsed);
    return {
      ...finalized,
      diagnostics: null,
      queryText: payload.queryText || null,
    };
  }

  if (!payload.queryText) {
    throw new Error("Missing `queryText` for text inspection.");
  }

  const result = await runTextEmbedding(payload.queryText, {
    includeDiagnostics: true,
  });
  const finalized = finalizeEmbeddingResult(result.vector, MODEL_ID);

  return {
    ...finalized,
    diagnostics: result.diagnostics,
    queryText: payload.queryText,
  };
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
  inspectTextEmbedding,
  resolveImageSearchEmbedding,
  resolveStoredEmbedding,
  resolveTextSearchEmbedding,
};
