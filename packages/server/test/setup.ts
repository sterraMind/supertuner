/**
 * Node may warn that `node:sqlite` is experimental. The warning is noise in
 * the test output (the API is stable for our use), so it is filtered here and
 * only here — production keeps the warning visible.
 */
const original = process.emitWarning.bind(process);

process.emitWarning = ((warning: string | Error, ...rest: unknown[]): void => {
  const name = typeof rest[0] === "string" ? rest[0] : undefined;
  const text = typeof warning === "string" ? warning : warning.message;
  if (name === "ExperimentalWarning" && /sqlite/i.test(text)) return;
  original(warning, ...(rest as Parameters<typeof original>));
}) as typeof process.emitWarning;
