import { describe, expect, it } from "vitest";
import {
  DARK_QUERY,
  THEME_STORAGE_KEY,
  applyTheme,
  createThemeController,
  isThemePref,
  readThemePref,
  resolveTheme,
  writeThemePref,
  type MediaQueryLike,
} from "./theme";

function memoryStorage(initial: Record<string, string> = {}): Storage {
  const map = new Map<string, string>(Object.entries(initial));
  return {
    get length() {
      return map.size;
    },
    clear: () => map.clear(),
    getItem: (key: string) => map.get(key) ?? null,
    key: (index: number) => [...map.keys()][index] ?? null,
    removeItem: (key: string) => {
      map.delete(key);
    },
    setItem: (key: string, value: string) => {
      map.set(key, value);
    },
  } as Storage;
}

function fakeDocument() {
  const attributes: Record<string, string> = {};
  const style: Record<string, string> = {};
  const doc = {
    documentElement: {
      setAttribute: (name: string, value: string) => {
        attributes[name] = value;
      },
      getAttribute: (name: string) => attributes[name] ?? null,
      style,
    },
  } as unknown as Document;
  return { doc, attributes, style };
}

function fakeMedia(dark: boolean) {
  const listeners = new Set<(event: { matches: boolean }) => void>();
  const media: MediaQueryLike & { emit(): void; matches: boolean } = {
    matches: dark,
    addEventListener: (_type, listener) => {
      listeners.add(listener);
    },
    removeEventListener: (_type, listener) => {
      listeners.delete(listener);
    },
    emit() {
      for (const listener of listeners) listener({ matches: media.matches });
    },
  };
  return media;
}

describe("theme preference storage", () => {
  it("treats an empty storage as system", () => {
    expect(readThemePref(memoryStorage())).toBe("system");
    expect(readThemePref(null)).toBe("system");
  });

  it("ignores junk values and keeps valid ones", () => {
    expect(readThemePref(memoryStorage({ [THEME_STORAGE_KEY]: "purple" }))).toBe("system");
    expect(readThemePref(memoryStorage({ [THEME_STORAGE_KEY]: "dark" }))).toBe("dark");
    expect(isThemePref("system")).toBe(true);
    expect(isThemePref("auto")).toBe(false);
  });

  it("round-trips a written preference", () => {
    const storage = memoryStorage();
    writeThemePref(storage, "light");
    expect(storage.getItem(THEME_STORAGE_KEY)).toBe("light");
    expect(readThemePref(storage)).toBe("light");
  });

  it("survives a storage that throws", () => {
    const hostile = {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("denied");
      },
    } as unknown as Storage;
    expect(readThemePref(hostile)).toBe("system");
    expect(() => writeThemePref(hostile, "dark")).not.toThrow();
  });
});

describe("resolveTheme", () => {
  it("honours an explicit preference over the system value", () => {
    expect(resolveTheme("light", true)).toBe("light");
    expect(resolveTheme("dark", false)).toBe("dark");
  });

  it("follows the system value when the preference is system", () => {
    expect(resolveTheme("system", true)).toBe("dark");
    expect(resolveTheme("system", false)).toBe("light");
  });
});

describe("theme controller", () => {
  it("applies the persisted theme on construction", () => {
    const { doc, attributes } = fakeDocument();
    const controller = createThemeController({
      storage: memoryStorage({ [THEME_STORAGE_KEY]: "dark" }),
      document: doc,
      matchMedia: () => fakeMedia(false),
    });
    expect(controller.pref).toBe("dark");
    expect(controller.theme).toBe("dark");
    expect(attributes["data-theme"]).toBe("dark");
    expect(doc.documentElement.style.colorScheme).toBe("dark");
  });

  it("persists a change and updates the document", () => {
    const { doc, attributes } = fakeDocument();
    const storage = memoryStorage();
    const controller = createThemeController({ storage, document: doc, matchMedia: () => fakeMedia(false) });
    expect(attributes["data-theme"]).toBe("light");

    controller.set("dark");
    expect(controller.pref).toBe("dark");
    expect(storage.getItem(THEME_STORAGE_KEY)).toBe("dark");
    expect(attributes["data-theme"]).toBe("dark");
  });

  it("follows the system preference live only while on system", () => {
    const { doc, attributes } = fakeDocument();
    const media = fakeMedia(false);
    const controller = createThemeController({ storage: memoryStorage(), document: doc, matchMedia: () => media });

    expect(controller.theme).toBe("light");
    media.matches = true;
    media.emit();
    expect(controller.theme).toBe("dark");
    expect(attributes["data-theme"]).toBe("dark");

    controller.set("light");
    media.matches = false;
    media.emit();
    expect(controller.theme).toBe("light");

    controller.set("system");
    media.matches = true;
    media.emit();
    expect(controller.theme).toBe("dark");
  });

  it("notifies subscribers and stops after dispose", () => {
    let calls = 0;
    const controller = createThemeController({
      storage: memoryStorage(),
      document: fakeDocument().doc,
      matchMedia: () => fakeMedia(false),
    });
    const unsubscribe = controller.subscribe(() => {
      calls += 1;
    });
    controller.set("dark");
    expect(calls).toBe(1);
    unsubscribe();
    controller.set("light");
    expect(calls).toBe(1);
  });

  it("uses the documented media query and applies data-theme directly", () => {
    expect(DARK_QUERY).toBe("(prefers-color-scheme: dark)");
    const { doc, attributes } = fakeDocument();
    applyTheme("dark", doc);
    expect(attributes["data-theme"]).toBe("dark");
  });
});