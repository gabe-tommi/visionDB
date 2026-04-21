const { getApps, initializeApp } = require("firebase-admin/app");
const { getAuth } = require("firebase-admin/auth");
const { getDataConnect } = require("firebase-admin/data-connect");

/*
 * Small Firebase bootstrap module.
 *
 * This file centralizes all Firebase Admin initialization so the rest of the
 * backend can ask for Auth or SQL Connect clients without worrying about
 * app lifecycle, duplicate initialization, or environment variable plumbing.
 */
function requireEnv(name) {
  const value = process.env[name];

  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }

  return value;
}

/*
 * Returns the singleton Firebase Admin app for this Node process.
 *
 * Firebase Admin should only be initialized once. If the app already exists,
 * we reuse it. If not, we create it using the project ID from env when present.
 */
function getFirebaseApp() {
  if (!getApps().length) {
    const options = {};

    if (process.env.FIREBASE_PROJECT_ID) {
      options.projectId = process.env.FIREBASE_PROJECT_ID;
    }

    initializeApp(options);
  }

  return getApps()[0];
}

/*
 * Exposes Firebase Auth for token verification in Express middleware.
 */
function getFirebaseAuth() {
  return getAuth(getFirebaseApp());
}

/*
 * Builds the SQL Connect client used by repository code.
 *
 * These connector details must match the SQL Connect service you created in
 * Firebase. We keep the values in env so local/dev/prod can point at different
 * services without code changes.
 */
function getSqlConnect() {
  return getDataConnect(
    {
      location: requireEnv("FIREBASE_DATACONNECT_LOCATION"),
      serviceId: requireEnv("FIREBASE_DATACONNECT_SERVICE_ID"),
      connector: process.env.FIREBASE_DATACONNECT_CONNECTOR || "default",
    },
    getFirebaseApp()
  );
}

module.exports = {
  getFirebaseApp,
  getFirebaseAuth,
  getSqlConnect,
};
