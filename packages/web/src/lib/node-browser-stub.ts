/**
 * Browser stand-in for the Node built-ins that `@supertuner/core/config.ts`
 * imports. That module is Node-only (it derives paths from `SUPERTUNER_*`), but
 * the core barrel re-exports it; aliasing the built-ins here keeps the browser
 * graph resolvable while tree-shaking drops the unused config code.
 *
 * Nothing in the UI may call these. They exist so the module can be parsed.
 */

export const existsSync = (): boolean => false;
export const homedir = (): string => "/";
export const dirname = (path: string): string => path.replace(/[/\\][^/\\]*$/, "") || "/";
export const isAbsolute = (path: string): boolean => /^([A-Za-z]:[\\/]|\/)/.test(path);
export const join = (...parts: string[]): string => parts.filter(Boolean).join("/");
export const resolve = (...parts: string[]): string => parts.filter(Boolean).join("/");
export const fileURLToPath = (url: string | URL): string => String(url).replace(/^file:\/\//, "");