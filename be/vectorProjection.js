const { EMBEDDING_DIMENSION } = require("./embeddingService");

const DEFAULT_METHOD = "pca";
const SUPPORTED_METHODS = new Set(["pca", "tsne", "umap"]);

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeSimilarity(distance) {
  const numericDistance = Number(distance);

  if (Number.isNaN(numericDistance)) {
    return null;
  }

  const similarity = 1 - numericDistance;
  return clamp(similarity, 0, 1);
}

function extractVector(embedding) {
  if (!embedding || !Array.isArray(embedding.vector)) {
    return null;
  }

  if (embedding.vector.length !== EMBEDDING_DIMENSION) {
    return null;
  }

  return embedding.vector.map((value) => Number(value));
}

function pickEmbedding(image) {
  if (!image || !Array.isArray(image.embeddings_on_image)) {
    return null;
  }

  return image.embeddings_on_image.find((embedding) => Array.isArray(embedding.vector)) || null;
}

function normalizeMethod(method) {
  if (!method) {
    return DEFAULT_METHOD;
  }

  const normalized = String(method).trim().toLowerCase();
  return SUPPORTED_METHODS.has(normalized) ? normalized : DEFAULT_METHOD;
}

function projectSmallSet(vectors) {
  if (!vectors.length) {
    return [];
  }

  if (vectors.length === 1) {
    return [[0, 0]];
  }

  return vectors.map((_vector, index) => {
    const x = index === 0 ? -1 : 1;
    return [x, 0];
  });
}

async function projectWithPca(vectors) {
  if (vectors.length < 3) {
    return projectSmallSet(vectors);
  }

  const { PCA } = require("ml-pca");
  const nComponents = Math.min(2, vectors.length - 1, vectors[0]?.length || 0);

  if (nComponents < 1) {
    return projectSmallSet(vectors);
  }

  const pca = new PCA(vectors, { center: true, scale: false });
  return pca.predict(vectors, { nComponents }).to2DArray().map((coords) => [
    coords[0] || 0,
    coords[1] || 0,
  ]);
}

async function projectWithTsne(vectors, options = {}) {
  if (vectors.length < 3) {
    return projectSmallSet(vectors);
  }

  const TSNE = require("tsne-js");
  const total = vectors.length;
  const perplexity = clamp(Math.floor((total - 1) / 3), 1, Math.max(1, total - 1));
  const iterations = options.iterations || 500;
  const tsne = new TSNE({
    dim: 2,
    perplexity,
    earlyExaggeration: 4,
    learningRate: options.learningRate || 100,
    nIter: iterations,
  });

  tsne.init({ data: vectors, type: "dense" });
  tsne.run();

  return tsne.getOutputScaled ? tsne.getOutputScaled() : tsne.getOutput();
}

async function projectWithUmap(vectors, options = {}) {
  if (vectors.length < 3) {
    return projectSmallSet(vectors);
  }

  const module = await import("umap-js");
  const UMAP = module.UMAP || module.default || module;
  const umap = new UMAP({
    nComponents: 2,
    nNeighbors:
      options.nNeighbors ||
      clamp(Math.floor(vectors.length / 2), 2, Math.max(2, vectors.length - 1)),
    minDist: options.minDist || 0.15,
  });

  return umap.fit(vectors);
}

async function projectVectors(vectors, method) {
  const normalizedMethod = normalizeMethod(method);

  if (vectors.length < 3 && normalizedMethod !== "pca") {
    return projectWithPca(vectors);
  }

  switch (normalizedMethod) {
    case "tsne":
      return projectWithTsne(vectors);
    case "umap":
      return projectWithUmap(vectors);
    case "pca":
    default:
      return projectWithPca(vectors);
  }
}

function buildProjectionPoints({ queryVector, queryLabel, matches }) {
  const items = [];

  items.push({
    id: "query",
    label: queryLabel || "Query",
    kind: "query",
    similarity: 1,
    distance: 0,
    vector: queryVector,
  });

  matches.forEach((match) => {
    const image = match.image || match;
    const embedding = pickEmbedding(image);
    const vector = extractVector(embedding);

    if (!vector) {
      return;
    }

    const distance = match?._metadata?.distance;

    items.push({
      id: image.id,
      label: image.description || "Untitled",
      kind: "match",
      similarity: normalizeSimilarity(distance),
      distance,
      vector,
      image: {
        id: image.id,
        imageUrl: image.imageUrl,
        description: image.description,
      },
      embeddingId: embedding?.id || null,
    });
  });

  return items;
}

module.exports = {
  buildProjectionPoints,
  normalizeMethod,
  projectVectors,
  SUPPORTED_METHODS,
};
