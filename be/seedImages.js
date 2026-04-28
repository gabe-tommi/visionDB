const fs = require("fs");
const path = require("path");
const { getStorageBucket } = require("./firebaseAdmin");
const { resolveStoredEmbedding } = require("./embeddingService");
const {
  createImageRecord,
  listImages,
  upsertEmbeddingForImage,
} = require("./visionDbRepository");

const SAMPLE_IMAGES_DIR = path.join(__dirname, "sampleImages");

/*
 * Uploads a local image file to Firebase Storage and returns its public
 * download URL.
 */
async function uploadToFirebase(filePath, filename) {
  const bucket = getStorageBucket();
  const destination = `sampleImages/${filename}`;
  const file = bucket.file(destination);

  await bucket.upload(filePath, {
    destination,
    metadata: { contentType: "image/jpeg" },
  });

  // Make the file publicly readable and return its URL.
  await file.makePublic();
  return `https://storage.googleapis.com/${bucket.name}/${destination}`;
}

/*
 * Reads all .JPEG/.jpg files from the sampleImages directory and seeds them
 * into the database.  Already-present URLs are skipped so re-runs are safe.
 *
 * Strategy:
 *  1. List existing images from the DB (up to the total sample count).
 *  2. Build a Set of imageUrls already stored.
 *  3. For each local file not yet in the DB: upload → embed → insert.
 */
async function seedSampleImages() {
  if (!fs.existsSync(SAMPLE_IMAGES_DIR)) {
    console.warn(`[seed] sampleImages directory not found at ${SAMPLE_IMAGES_DIR} — skipping.`);
    return;
  }

  const files = fs
    .readdirSync(SAMPLE_IMAGES_DIR)
    .filter((f) => /\.(jpe?g|png|webp)$/i.test(f))
    .sort();

  if (files.length === 0) {
    console.log("[seed] No image files found in sampleImages — skipping.");
    return;
  }

  // Fetch enough rows to cover all possible seeds.
  const existing = await listImages(files.length + 1);
  const existingUrls = new Set(existing.map((img) => img.imageUrl));

  const bucket = getStorageBucket();
  const missing = files.filter((filename) => {
    const destination = `sampleImages/${filename}`;
    const expectedUrl = `https://storage.googleapis.com/${bucket.name}/${destination}`;
    return !existingUrls.has(expectedUrl);
  });

  if (missing.length === 0) {
    console.log("[seed] All sample images already in DB — skipping.");
    return;
  }

  console.log(`[seed] Uploading ${missing.length} of ${files.length} sample image(s)…`);

  let succeeded = 0;
  let failed = 0;

  for (const filename of missing) {
    const filePath = path.join(SAMPLE_IMAGES_DIR, filename);
    try {
      const imageUrl = await uploadToFirebase(filePath, filename);

      const embedding = await resolveStoredEmbedding({ imageUrl });
    // for image table updates
      const created = await createImageRecord({
        imageUrl,
        description: path.basename(filename, path.extname(filename)).replace(/_/g, " "),
        createdAt: new Date().toISOString(),
      });
    // for embedding table updates
      await upsertEmbeddingForImage(created.id, {
        vector: embedding.vector,
        dimension: embedding.dimension,
        modelUsed: embedding.modelUsed,
        generatedAt: new Date().toISOString(),
      });

      succeeded++;
      if (succeeded % 50 === 0) {
        console.log(`[seed] ${succeeded}/${missing.length} done…`);
      }
    } catch (err) {
      failed++;
      console.error(`[seed] ✗ ${filename}: ${err.message}`);
    }
  }

  console.log(`[seed] Done. ${succeeded} uploaded, ${failed} failed.`);
}

module.exports = { seedSampleImages };

