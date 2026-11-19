import { useCallback, useEffect, useState } from "react";

/**
 * Theme preference is one of three values; `system` follows
 * `prefers-color-scheme` live. The resolved theme is written to
 * `<html data-theme>` so tokens.css can do the rest.
 *
 * index.html carries a tiny inline copy of the read/resolve step so the first
 * paint is already correct; keep both in sync via THEME_STORAGE_KEY.
 */

export type ThemePref = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

export const THEME_STORAGE_KEY = "supertuner.theme";
export const DARK_QUERY = "(prefers-color-scheme: dark)";

const PREFS: readonly ThemePref[] = ["system", "light", "dark"];

export function isThemePref(value: unknown): value is ThemePref {
  return typeof value === "string" && (PREFS as readonly string[]).includes(value);
}

/** Reads the persisted preference. Storage failures and junk fall back to system. */
export function readThemePref(storage: Storage | null | undefined): ThemePref {
  if (!storage) return "system";
  try {
    const raw = storage.getItem(THEME_STORAGE_KEY);
    return isThemePref(raw) ? raw : "system";
  } catch {
    return "system";
  }
}

export function writeThemePref(storage: Storage | null | undefined, pref: ThemePref): void {
  if (!storage) return;
  try {
    storage.setItem(THEME_STORAGE_KEY, pref);
  } catch {
    /* private mode / quota: the in-memory preference still applies. */
  }
}

export interface MediaQueryLike {
  matches: boolean;
  addEventListener?: (type: "change", listener: (event: { matches: boolean }) => void) => void;
  removeEventListener?: (type: "change", listener: (event: { matches: boolean }) => void) => void;
  addListener?: (listener: (event: { matches: boolean }) => void) => void;
  removeListener?: (listener: (event: { matches: boolean }) => void) => void;
}

export function resolveTheme(pref: ThemePref, systemPrefersDark: boolean): ResolvedTheme {
  if (pref === "light" || pref === "dark") return pref;
  return systemPrefersDark ? "dark" : "light";
}

export function applyTheme(theme: ResolvedTheme, doc: Document = document): void {
  doc.documentElement.setAttribute("data-theme", theme);
  doc.documentElement.style.colorScheme = theme;
}

function safeStorage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}

function defaultMatchMedia(query: string): MediaQueryLike | null {
  try {
    return typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? (window.matchMedia(query) as MediaQueryLike)
      : null;
  } catch {
    return null;
  }
}

function subscribe(
  media: MediaQueryLike | null,
  listener: (event: { matches: boolean }) => void,
): () => void {
  if (!media) return () => {};
  if (media.addEventListener) {
    media.addEventListener("change", listener);
    return () => media.removeEventListener?.("change", listener);
  }
  if (media.addListener) {
    media.addListener(listener);
    return () => media.removeListener?.(listener);
  }
  return () => {};
}

export interface ThemeEnv {
  storage?: Storage | null;
  document?: Document;
  matchMedia?: (query: string) => MediaQueryLike | null;
}

export interface ThemeController {
  readonly pref: ThemePref;
  readonly theme: ResolvedTheme;
  set(pref: ThemePref): void;
  /** Re-evaluate `system` without changing the preference. */
  sync(): void;
  dispose(): void;
  subscribe(listener: () => void): () => void;
}

/** Build a controller (plain object) so tests can drive it without React. */
export function createThemeController(env: ThemeEnv = {}): ThemeController {
  const doc = env.document ?? (typeof document === "undefined" ? null : document);
  const storage = env.storage === undefined ? safeStorage() : env.storage;
  const media = (env.matchMedia ?? defaultMatchMedia)(DARK_QUERY);
  const listeners = new Set<() => void>();

  let pref = readThemePref(storage);
  let lastApplied: ResolvedTheme | null = null;

  const current = (): ResolvedTheme => resolveTheme(pref, media?.matches === true);

  const apply = (): void => {
    const theme = current();
    if (theme !== lastApplied) {
      lastApplied = theme;
      if (doc) applyTheme(theme, doc);
    }
  };

  const notify = (): void => {
    apply();
    for (const listener of listeners) listener();
  };

  const unsubscribeMedia = subscribe(media, () => {
    if (pref === "system") notify();
  });

  apply();

  return {
    get pref() {
      return pref;
    },
    get theme() {
      return current();
    },
    set(next) {
      if (!isThemePref(next) || next === pref) return;
      pref = next;
      writeThemePref(storage, next);
      notify();
    },
    sync: notify,
    dispose() {
      unsubscribeMedia();
      listeners.clear();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export interface ThemeState {
  pref: ThemePref;
  theme: ResolvedTheme;
  setPref: (pref: ThemePref) => void;
}

/**
 * React binding. One controller per app; the first use installs the theme on
 * <html>, later mounts reuse the same controller module-wide.
 */
let sharedController: ThemeController | null = null;

export function getThemeController(): ThemeController {
  sharedController ??= createThemeController();
  return sharedController;
}

export function resetThemeControllerForTests(): void {
  sharedController?.dispose();
  sharedController = null;
}

export function useTheme(): ThemeState {
  const [controller] = useState(getThemeController);
  const [pref, setPrefState] = useState<ThemePref>(controller.pref);
  const [theme, setThemeState] = useState<ResolvedTheme>(controller.theme);

  useEffect(() => controller.subscribe(() => {
    setPrefState(controller.pref);
    setThemeState(controller.theme);
  }), [controller]);

  const setPref = useCallback(
    (next: ThemePref) => {
      controller.set(next);
      setPrefState(controller.pref);
      setThemeState(controller.theme);
    },
    [controller],
  );

  return { pref, theme, setPref };
}