/**
 * Child-process helpers shared by the engine client and the run supervisor.
 *
 * Windows has no SIGTERM: `child.kill()` terminates only the direct process
 * and orphans grandchildren (`uv run` → Python → training worker). A hard kill
 * therefore goes through `taskkill /PID <pid> /T /F`, and only when the child
 * ignored the graceful path.
 */

import { spawn, type ChildProcess } from "node:child_process";

/** Resolve `true` when the child exits within `timeoutMs`, `false` otherwise. */
export function waitForExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve(true);
  return new Promise<boolean>((resolve) => {
    let settled = false;
    const finish = (exited: boolean): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.removeListener("exit", onExit);
      child.removeListener("close", onExit);
      resolve(exited);
    };
    const onExit = (): void => finish(true);
    const timer = setTimeout(() => finish(false), timeoutMs);
    timer.unref?.();
    child.once("exit", onExit);
    child.once("close", onExit);
  });
}

async function taskkillTree(pid: number): Promise<void> {
  await new Promise<void>((resolve) => {
    let child: ChildProcess;
    try {
      child = spawn("taskkill", ["/PID", String(pid), "/T", "/F"], {
        shell: false,
        windowsHide: true,
        stdio: "ignore",
      });
    } catch {
      resolve();
      return;
    }
    child.once("error", () => resolve());
    child.once("close", () => resolve());
  });
}

export interface TerminateOptions {
  /** Signal used on POSIX; ignored on Windows (there is no graceful signal). */
  signal?: NodeJS.Signals;
  /** How long to wait for a graceful exit before escalating. */
  graceMs?: number;
  /** Skip the graceful attempt on Windows and run taskkill straight away. */
  force?: boolean;
}

/** Stop a child and its descendants; resolves `true` if the child is gone. */
export async function terminateChildTree(child: ChildProcess, options: TerminateOptions = {}): Promise<boolean> {
  const { signal = "SIGTERM", graceMs = 5_000, force = false } = options;
  if (child.exitCode !== null || child.signalCode !== null) return true;
  const pid = child.pid;

  if (process.platform === "win32") {
    if (pid === undefined) return true;
    if (!force) {
      try {
        child.kill();
      } catch {
        return true;
      }
      if (await waitForExit(child, graceMs)) return true;
    }
    await taskkillTree(pid);
    return await waitForExit(child, graceMs);
  }

  try {
    child.kill(signal);
  } catch {
    return true;
  }
  if (await waitForExit(child, graceMs)) return true;
  try {
    child.kill("SIGKILL");
  } catch {
    return true;
  }
  return await waitForExit(child, 2_000);
}