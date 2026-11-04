/** Bounded public FPL cache. Failures are retried, never stored as empty data. */
const MAX_ENTRIES = 256;
const MAX_CONCURRENT = 4;
type Entry = { value: unknown; expires: number };
type Cache = { entries: Map<string, Entry>; pending: Map<string, Promise<unknown>>; active: number; queue: (() => void)[] };
const caches = new WeakMap<typeof fetch, Cache>();

export function createLeagueFetcher(fetcher: typeof fetch, budgetMs = 18_000) {
  let cache = caches.get(fetcher);
  if (!cache) {
    cache = { entries: new Map(), pending: new Map(), active: 0, queue: [] };
    caches.set(fetcher, cache);
  }
  const state = cache;
  const deadline = Date.now() + budgetMs;
  return async (path: string, finalized = false): Promise<unknown> => {
    const cached = state.entries.get(path);
    if (cached && cached.expires > Date.now()) return cached.value;
    if (state.pending.has(path)) return state.pending.get(path)!;
    const task = new Promise<unknown>((resolve, reject) => {
      const run = async () => {
        state.active += 1;
        try {
          const remaining = deadline - Date.now();
          if (remaining <= 0) throw new Error("League time budget exceeded");
          const response = await fetcher(`https://fantasy.premierleague.com/api/${path}`, {
            headers: { Accept: "application/json" }, cache: "no-store", redirect: "error",
            signal: AbortSignal.timeout(Math.max(1, Math.min(4_000, remaining))),
          });
          if (!response.ok) throw new Error("Official league data unavailable");
          let value: unknown = await response.json();
          // A live response is large. Diagnosis uses only finalized player scores.
          if (/^event\/\d+\/live\/$/.test(path)) {
            const raw = value as { elements?: { id: unknown; stats?: { total_points?: unknown } }[] };
            if (Array.isArray(raw?.elements)) value = { elements: raw.elements.map(p => ({ id: p.id, stats: { total_points: p.stats?.total_points } })) };
          }
          state.entries.delete(path);
          state.entries.set(path, { value, expires: Date.now() + (finalized ? 6 * 60 * 60_000 : 60_000) });
          while (state.entries.size > MAX_ENTRIES) state.entries.delete(state.entries.keys().next().value!);
          resolve(value);
        } catch (error) { reject(error); }
        finally { state.active -= 1; state.queue.shift()?.(); }
      };
      if (state.active < MAX_CONCURRENT) void run();
      else state.queue.push(() => { void run(); });
    });
    state.pending.set(path, task);
    try { return await task; }
    finally { state.pending.delete(path); }
  };
}
