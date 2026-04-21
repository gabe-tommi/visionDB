const { getSqlConnect } = require("./firebaseAdmin");

/*
 * Thin wrapper around SQL Connect execution.
 *
 * The repository layer only needs two things:
 * 1. a way to execute GraphQL against SQL Connect
 * 2. a consistent error shape if SQL Connect returns GraphQL errors
 *
 * Keeping that behavior here avoids repeating the same response/error handling
 * in every query and mutation helper.
 */
function formatGraphqlErrors(errors) {
  if (!Array.isArray(errors) || !errors.length) {
    return "Unknown SQL Connect error";
  }

  return errors
    .map((error) => error.message || JSON.stringify(error))
    .join("; ");
}

/*
 * Runs arbitrary GraphQL against SQL Connect.
 *
 * `readOnly` switches to SQL Connect's read-only execution API for queries.
 * For writes, the regular execution API is used instead.
 */
async function runAdminGraphql(query, variables, { readOnly = false } = {}) {
  const sqlConnect = getSqlConnect();
  const options = {};

  if (variables !== undefined) {
    options.variables = variables;
  }

  const response = readOnly
    ? await sqlConnect.executeGraphqlRead(query, options)
    : await sqlConnect.executeGraphql(query, options);

  if (response?.errors?.length) {
    throw new Error(formatGraphqlErrors(response.errors));
  }

  return response?.data || {};
}

module.exports = {
  runAdminGraphql,
};
