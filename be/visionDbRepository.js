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
  mutation CreateImage($data: Image_Data!) {
    image_insert(data: $data) {
      id
    }
  }
`;

const CREATE_EMBEDDING_MUTATION = `
  mutation CreateEmbedding($data: Embedding_Data!) {
    embedding_insert(data: $data) {
      id
    }
  }
`;

const UPDATE_IMAGE_MUTATION = `
  mutation UpdateImage($id: UUID!, $data: Image_Data!) {
    image_update(id: $id, data: $data) {
      id
    }
  }
`;

const UPDATE_EMBEDDING_MUTATION = `
  mutation UpdateEmbedding($id: UUID!, $data: Embedding_Data!) {
    embedding_update(id: $id, data: $data) {
      id
    }
  }
`;

const DELETE_IMAGE_MUTATION = `
  mutation DeleteImage($id: UUID!) {
    image_delete(id: $id) {
      id
    }
  }
`;

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

/*
 * Creates a single image row.
 */
async function createImageRecord(data) {
  const result = await runAdminGraphql(CREATE_IMAGE_MUTATION, { data });
  return result.image_insert;
}

/*
 * Creates a single embedding row.
 */
async function createEmbeddingRecord(data) {
  const result = await runAdminGraphql(CREATE_EMBEDDING_MUTATION, { data });
  return result.embedding_insert;
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
    {
      compare: vector,
      limit,
      within,
    },
    { readOnly: true }
  );

  return result.embeddings_vector_similarity || [];
}

/*
 * Applies partial updates to an existing image row.
 */
async function updateImageRecord(id, data) {
  const sanitized = stripUndefined(data);

  if (!Object.keys(sanitized).length) {
    return null;
  }

  const result = await runAdminGraphql(UPDATE_IMAGE_MUTATION, {
    id,
    data: sanitized,
  });

  return result.image_update;
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

  const result = await runAdminGraphql(UPDATE_EMBEDDING_MUTATION, {
    id: existingEmbedding.id,
    data: payload,
  });

  return result.embedding_update;
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

    if (result.image_delete?.id) {
      deletedIds.push(result.image_delete.id);
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
