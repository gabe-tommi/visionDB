const { runAdminGraphql } = require("./sqlConnect");

/*
 * Repository layer for the app's SQL Connect schema.
 *
 * This module is the only place that knows the actual GraphQL operations used
 * to talk to your `Image` and `Embedding` tables. The Express routes call these
 * helpers instead of embedding raw GraphQL strings directly in route handlers.
 *
 * Current assumptions based on your schema:
 * - `Image` has an implicit UUID `id`
 * - `Embedding` has an implicit UUID `id`
 * - `Embedding.image: Image!` creates an implicit foreign key `imageId`
 * - reverse traversal from `Image` to `Embedding` is exposed as
 *   `embeddings_on_image`
 */
const CREATE_IMAGE_MUTATION = `
  mutation CreateImage(
    $imageUrl: String!
    $description: String
    $tags: [String!]
    $metadata: String
    $createdAt: Timestamp
  ) {
    image_insert(data: {
      imageUrl: $imageUrl
      description: $description
      tags: $tags
      metadata: $metadata
      createdAt: $createdAt
    })
  }
`;

const CREATE_EMBEDDING_MUTATION = `
  mutation CreateEmbedding(
    $imageId: UUID!
    $vector: Vector
    $dimension: Int!
    $modelUsed: String!
    $generatedAt: Timestamp
  ) {
    embedding_insert(data: {
      imageId: $imageId
      vector: $vector
      dimension: $dimension
      modelUsed: $modelUsed
      generatedAt: $generatedAt
    })
  }
`;

const DELETE_IMAGE_MUTATION = `
  mutation DeleteImage($id: UUID!) {
    image_delete(id: $id)
  }
`;

const IMAGE_UPDATE_VARIABLE_TYPES = {
  imageUrl: "String",
  description: "String",
  tags: "[String!]",
  metadata: "String",
  createdAt: "Timestamp",
};

const EMBEDDING_UPDATE_VARIABLE_TYPES = {
  imageId: "UUID",
  vector: "Vector",
  dimension: "Int",
  modelUsed: "String",
  generatedAt: "Timestamp",
};

const GET_IMAGE_QUERY = `
  query GetImage($id: UUID!) {
    image(id: $id) {
      id
      imageUrl
      description
      tags
      metadata
      createdAt
      embeddings_on_image {
        id
        vector
        dimension
        modelUsed
        generatedAt
      }
    }
  }
`;

const LIST_IMAGES_QUERY = `
  query ListImages($limit: Int!) {
    images(limit: $limit) {
      id
      imageUrl
      description
      tags
      metadata
      createdAt
      embeddings_on_image {
        id
        vector
        dimension
        modelUsed
        generatedAt
      }
    }
  }
`;

const SEARCH_IMAGES_BY_VECTOR_QUERY = `
  query SearchImagesByVector($compare: Vector!, $limit: Int!, $within: Float) {
    embeddings_vector_similarity(
      compare: $compare
      limit: $limit
      method: COSINE
      within: $within
    ) {
      id
      dimension
      modelUsed
      generatedAt
      image {
        id
        imageUrl
        description
        tags
        metadata
        createdAt
        embeddings_on_image {
          id
          vector
          dimension
          modelUsed
          generatedAt
        }
      }
      _metadata {
        distance
      }
    }
  }
`;

/*
 * Utility to remove fields that were not provided by the caller.
 *
 * This matters for update operations because SQL Connect should only receive
 * fields we actually intend to change.
 */
function stripUndefined(object) {
  return Object.fromEntries(
    Object.entries(object).filter(([, value]) => value !== undefined)
  );
}

function buildUpdateMutation({
  operationName,
  fieldName,
  variableTypes,
  data,
}) {
  const fields = Object.keys(data);
  const variableDefinitions = [
    "$id: UUID!",
    ...fields.map((field) => `$${field}: ${variableTypes[field]}`),
  ].join(", ");
  const dataFields = fields
    .map((field) => `${field}: $${field}`)
    .join("\n      ");

  return `
    mutation ${operationName}(${variableDefinitions}) {
      ${fieldName}(id: $id, data: {
        ${dataFields}
      })
    }
  `;
}

/*
 * SQL Connect key-return mutations expose generated KeyOutput values directly,
 * not selectable objects. Depending on SDK serialization, the key can arrive
 * as a UUID string or as an object containing the primary key.
 */
function normalizeKeyOutput(value) {
  if (typeof value === "string") {
    return { id: value };
  }

  if (value && typeof value === "object") {
    if (typeof value.id === "string") {
      return { id: value.id };
    }

    const firstStringValue = Object.values(value).find((entry) => {
      return typeof entry === "string";
    });

    if (firstStringValue) {
      return { id: firstStringValue };
    }

    const nestedKey = Object.values(value)
      .map((entry) => normalizeKeyOutput(entry))
      .find((entry) => entry?.id);

    if (nestedKey) {
      return nestedKey;
    }
  }

  return value;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeDistance(value) {
  const distance = Number(value);
  return Number.isFinite(distance) ? distance : null;
}

function addSimilarityScore(match) {
  const distance = normalizeDistance(match?._metadata?.distance);

  if (distance === null) {
    return match;
  }

  return {
    ...match,
    distance,
    similarity: clamp(1 - distance, 0, 1),
  };
}

/*
 * Creates a single image row.
 */
async function createImageRecord(data) {
  const result = await runAdminGraphql(CREATE_IMAGE_MUTATION, data);
  return normalizeKeyOutput(result.image_insert);
}

/*
 * Creates a single embedding row.
 */
async function createEmbeddingRecord(data) {
  const result = await runAdminGraphql(CREATE_EMBEDDING_MUTATION, data);
  return normalizeKeyOutput(result.embedding_insert);
}

/*
 * Loads one image and its related embeddings so route handlers can return a
 * single fully-hydrated object after create/update flows.
 */
async function getImageById(id) {
  const result = await runAdminGraphql(GET_IMAGE_QUERY, { id }, { readOnly: true });
  return result.image;
}

/*
 * Returns a page of images with their embeddings.
 */
async function listImages(limit = 50) {
  const result = await runAdminGraphql(
    LIST_IMAGES_QUERY,
    { limit },
    { readOnly: true }
  );

  return result.images || [];
}

/*
 * Runs pgvector similarity search through SQL Connect's generated
 * `*_similarity` field for the `Embedding.vector` column.
 */
async function searchImagesByVector({ vector, limit = 5, within }) {
  const result = await runAdminGraphql(
    SEARCH_IMAGES_BY_VECTOR_QUERY,
    stripUndefined({
      compare: vector,
      limit,
      within,
    }),
    { readOnly: true }
  );

  return (result.embeddings_vector_similarity || []).map(addSimilarityScore);
}

/*
 * Applies partial updates to an existing image row.
 */
async function updateImageRecord(id, data) {
  const sanitized = stripUndefined(data);

  if (!Object.keys(sanitized).length) {
    return null;
  }

  const mutation = buildUpdateMutation({
    operationName: "UpdateImage",
    fieldName: "image_update",
    variableTypes: IMAGE_UPDATE_VARIABLE_TYPES,
    data: sanitized,
  });

  const result = await runAdminGraphql(mutation, {
    id,
    ...sanitized,
  });

  return normalizeKeyOutput(result.image_update);
}

/*
 * Keeps the current one-image-to-first-embedding behavior simple.
 *
 * If the image has no embedding yet, create one.
 * If it already has one, update the first related embedding row.
 *
 * If you later decide to support multiple embeddings per image, this helper is
 * the main place that behavior would need to change.
 */
async function upsertEmbeddingForImage(imageId, data) {
  const image = await getImageById(imageId);

  if (!image) {
    return null;
  }

  const existingEmbedding = image.embeddings_on_image?.[0];
  const payload = stripUndefined({
    imageId,
    vector: data.vector,
    dimension: data.dimension,
    modelUsed: data.modelUsed,
    generatedAt: data.generatedAt,
  });

  if (!existingEmbedding) {
    return createEmbeddingRecord(payload);
  }

  const mutation = buildUpdateMutation({
    operationName: "UpdateEmbedding",
    fieldName: "embedding_update",
    variableTypes: EMBEDDING_UPDATE_VARIABLE_TYPES,
    data: payload,
  });

  const result = await runAdminGraphql(mutation, {
    id: existingEmbedding.id,
    ...payload,
  });

  return normalizeKeyOutput(result.embedding_update);
}

/*
 * Deletes images one at a time and returns the IDs that were actually deleted.
 *
 * Since `Embedding.image` is required in your schema, SQL Connect/Postgres
 * should cascade-delete related embeddings automatically.
 */
async function deleteImagesByIds(ids) {
  const deletedIds = [];

  for (const id of ids) {
    const result = await runAdminGraphql(DELETE_IMAGE_MUTATION, { id });
    const deleted = normalizeKeyOutput(result.image_delete);

    if (deleted?.id) {
      deletedIds.push(deleted.id);
    }
  }

  return deletedIds;
}

module.exports = {
  createEmbeddingRecord,
  createImageRecord,
  deleteImagesByIds,
  getImageById,
  listImages,
  searchImagesByVector,
  updateImageRecord,
  upsertEmbeddingForImage,
};
