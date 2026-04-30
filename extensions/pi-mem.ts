/**
 * Pi-Mem — claude-mem extension for pi-mono agents
 *
 * Gives pi-agents (pi-coding-agent, custom pi-mono runtimes) persistent
 * cross-session memory by connecting to the claude-mem worker HTTP API.
 *
 * Derived from the OpenClaw plugin (claude-mem/openclaw/src/index.ts) which
 * is a proven integration pattern for pi-mono-based runtimes.
 *
 * v0.5.0 — Fork with all upstream features:
 *   - Observation buffering (local queue when worker down)
 *   - Worker auto-start (spawn worker if unreachable)
 *   - Context config (env var for observation count)
 *   - Viewer command (/memory-viewer)
 *   - Enhanced search (filters on memory_recall)
 *   - User prompt storage (passed to worker for indexing)
 *   - Worker health monitoring (periodic check)
 *   - Mode system (context-aware observation capture)
 *   - Progressive disclosure (index + get_observations tool)
 *   - Corpus management (build, list, get, query corpora)
 *
 * Install:
 *   pi install npm:pi-agent-memory
 *   — or —
 *   pi install git:github.com/HaqimIskandar/claude-mem --extensions pi-agent/extensions
 *
 * Requires: claude-mem worker running on localhost:37777
 */

import { Type } from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { basename, join } from "node:path";
import { spawn } from "node:child_process";

// =============================================================================
// Configuration
// =============================================================================

const DEFAULT_WORKER_PORT = 37777;

function discoverWorkerHost(): string {
  if (process.env.CLAUDE_MEM_HOST) return process.env.CLAUDE_MEM_HOST;
  const settingsDir = process.env.CLAUDE_MEM_DATA_DIR || join(homedir(), ".claude-mem");
  const settingsPath = join(settingsDir, "settings.json");
  if (existsSync(settingsPath)) {
    try {
      const settings = JSON.parse(readFileSync(settingsPath, "utf-8"));
      if (typeof settings.CLAUDE_MEM_WORKER_HOST === "string") return settings.CLAUDE_MEM_WORKER_HOST;
    } catch { /* ignore parse/read errors */ }
  }
  return "127.0.0.1";
}

function discoverWorkerPort(): number {
  if (process.env.CLAUDE_MEM_PORT) {
    const parsed = parseInt(process.env.CLAUDE_MEM_PORT, 10);
    if (Number.isFinite(parsed)) return parsed;
  }
  const settingsDir = process.env.CLAUDE_MEM_DATA_DIR || join(homedir(), ".claude-mem");
  const settingsPath = join(settingsDir, "settings.json");
  if (existsSync(settingsPath)) {
    try {
      const settings = JSON.parse(readFileSync(settingsPath, "utf-8"));
      if (typeof settings.CLAUDE_MEM_WORKER_PORT === "number" && Number.isFinite(settings.CLAUDE_MEM_WORKER_PORT)) {
        return settings.CLAUDE_MEM_WORKER_PORT;
      }
    } catch { /* ignore parse/read errors */ }
  }
  return DEFAULT_WORKER_PORT;
}

function discoverSettings(): Record<string, unknown> {
  const settingsDir = process.env.CLAUDE_MEM_DATA_DIR || join(homedir(), ".claude-mem");
  const settingsPath = join(settingsDir, "settings.json");
  if (existsSync(settingsPath)) {
    try {
      return JSON.parse(readFileSync(settingsPath, "utf-8"));
    } catch { /* ignore */ }
  }
  return {};
}

const WORKER_PORT = discoverWorkerPort();
const WORKER_HOST = discoverWorkerHost();
const SETTINGS = discoverSettings();
const PLATFORM_SOURCE = "pi-agent";
const MAX_TOOL_RESPONSE_LENGTH = 1000;
const SESSION_COMPLETE_DELAY_MS = 3000;
const WORKER_FETCH_TIMEOUT_MS = 10_000;
const MAX_SEARCH_LIMIT = 100;

// Gap 3: Context config — configurable observation count
const CONTEXT_OBSERVATION_COUNT = parseInt(
  process.env.PI_MEM_CONTEXT_OBSERVATIONS || String(SETTINGS.CLAUDE_MEM_CONTEXT_OBSERVATIONS || "50"),
  10
);

// Gap 1: Observation buffer location
// Dedup: per-tool cooldown window — suppress rapid-fire observations from same tool
const DEDUP_WINDOW_MS = parseInt(
  process.env.PI_MEM_DEDUP_WINDOW_MS || String(SETTINGS.CLAUDE_MEM_DEDUP_WINDOW_MS || "30000"),
 10
);
const recentToolSends = new Map<string, number>();

const BUFFER_DIR = join(homedir(), ".claude-mem", "pi-mem-buffer");
const BUFFER_FLUSH_INTERVAL_MS = 5_000;
const MAX_BUFFER_SIZE = 500;

// Gap 7: Health monitoring
const HEALTH_CHECK_INTERVAL_MS = 30_000;
let workerHealthy = true;

// Dead-man's switch: warn if no new observations stored for this long
const STALE_OBSERVATION_THRESHOLD_MS = 24 * 60 * 60 * 1000; // 24 hours

// =============================================================================
// Mode System — context-aware observation capture
//
// Modes define which tools to capture observations from and how to tag them.
// Configured via CLAUDE_MEM_MODE setting or PI_MEM_MODE env var.
// =============================================================================

interface CaptureMode {
  name: string;
  description: string;
  captureTools: string[]; // empty = capture ALL tools
  skipTools: string[];
  observationTypes: string[];
}

const MODES: Record<string, CaptureMode> = {
  code: {
    name: "Code Development",
    description: "Software development and engineering work",
    captureTools: [],
    skipTools: ["ListMcpResourcesTool", "SlashCommand", "Skill", "TodoWrite", "AskUserQuestion", "memory_recall", "get_observations"],
    observationTypes: ["bugfix", "feature", "refactor", "change", "discovery", "decision"],
  },
  plan: {
    name: "Planning",
    description: "Architecture and planning sessions",
    captureTools: ["read", "write", "bash"],
    skipTools: ["ListMcpResourcesTool", "SlashCommand", "Skill", "TodoWrite", "AskUserQuestion", "memory_recall", "get_observations"],
    observationTypes: ["decision", "discovery", "change"],
  },
  research: {
    name: "Research",
    description: "Investigation and analysis work",
    captureTools: ["read", "bash", "brave-search"],
    skipTools: ["ListMcpResourcesTool", "SlashCommand", "Skill", "TodoWrite", "AskUserQuestion", "memory_recall", "get_observations"],
    observationTypes: ["discovery", "decision"],
  },
};

const activeMode: CaptureMode = (() => {
  const modeKey = process.env.PI_MEM_MODE || String(SETTINGS.CLAUDE_MEM_MODE || "code");
  return MODES[modeKey] || MODES.code;
})();

// =============================================================================
// HTTP Helpers
// =============================================================================

function workerUrl(path: string): string {
  return `http://${WORKER_HOST}:${WORKER_PORT}${path}`;
}

function createTimeoutController(): { controller: AbortController; clear: () => void } {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), WORKER_FETCH_TIMEOUT_MS);
  return { controller, clear: () => clearTimeout(timer) };
}

async function workerPost(path: string, body: Record<string, unknown>): Promise<Record<string, unknown> | null> {
  const { controller, clear } = createTimeoutController();
  try {
    const response = await fetch(workerUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      console.error(`[pi-mem] Worker POST ${path} returned ${response.status}`);
      return null;
    }
    return (await response.json()) as Record<string, unknown>;
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === "AbortError") {
      console.error(`[pi-mem] Worker POST ${path} timed out after ${WORKER_FETCH_TIMEOUT_MS}ms`);
    } else {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[pi-mem] Worker POST ${path} failed: ${message}`);
    }
    return null;
  } finally {
    clear();
  }
}

function workerPostFireAndForget(path: string, body: Record<string, unknown>): void {
  fetch(workerUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`[pi-mem] Worker POST ${path} failed: ${message}`);
  });
}

async function workerGetText(path: string): Promise<string | null> {
  const { controller, clear } = createTimeoutController();
  try {
    const response = await fetch(workerUrl(path), { signal: controller.signal });
    if (!response.ok) {
      console.error(`[pi-mem] Worker GET ${path} returned ${response.status}`);
      return null;
    }
    return await response.text();
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === "AbortError") {
      console.error(`[pi-mem] Worker GET ${path} timed out after ${WORKER_FETCH_TIMEOUT_MS}ms`);
    } else {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[pi-mem] Worker GET ${path} failed: ${message}`);
    }
    return null;
  } finally {
    clear();
  }
}

// =============================================================================
// Gap 2: Worker Auto-Start
//
// If the worker is unreachable, attempt to start it using the same command
// the claude-mem monitor.sh uses: bun worker-service.cjs --daemon
// =============================================================================

async function isWorkerReachable(): Promise<boolean> {
  try {
    const { controller, clear } = createTimeoutController();
    const response = await fetch(workerUrl("/api/health"), { signal: controller.signal });
    clear();
    return response.ok;
  } catch {
    return false;
  }
}

async function attemptWorkerStart(): Promise<boolean> {
  // Try multiple common worker locations
  const workerPaths = [
    join(homedir(), ".claude/plugins/marketplaces/thedotmack/plugin/scripts/worker-service.cjs"),
    join(homedir(), ".claude/plugins/marketplaces/thedotmack/plugin/scripts/worker-service.mjs"),
  ];

  for (const workerPath of workerPaths) {
    if (!existsSync(workerPath)) continue;

    const bunPath = join(homedir(), ".bun/bin/bun");
    const execCmd = existsSync(bunPath) ? bunPath : "bun";

    console.error(`[pi-mem] Attempting worker auto-start: ${execCmd} ${workerPath} --daemon`);

    return new Promise((resolve) => {
      const child = spawn(execCmd, [workerPath, "--daemon"], {
        detached: true,
        stdio: "ignore",
        env: { ...process.env },
      });
      child.unref();

      child.on("error", (err) => {
        console.error(`[pi-mem] Worker auto-start failed: ${err.message}`);
        resolve(false);
      });

      // Give the worker 3 seconds to start, then check health
      setTimeout(async () => {
        const reachable = await isWorkerReachable();
        if (reachable) {
          console.error("[pi-mem] Worker auto-start successful");
        }
        resolve(reachable);
      }, 3000);
    });
  }

  console.error("[pi-mem] No worker binary found for auto-start");
  return false;
}

// =============================================================================
// Gap 1: Observation Buffer
//
// When the worker is unreachable, queue observations to a local JSON file.
// On next successful worker contact, flush the buffer.
// =============================================================================

interface BufferedObservation {
  contentSessionId: string;
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_response: string;
  cwd: string;
  platformSource: string;
  buffered_at: string;
}

function ensureBufferDir(): void {
  if (!existsSync(BUFFER_DIR)) {
    mkdirSync(BUFFER_DIR, { recursive: true });
  }
}

function getBufferPath(): string {
  return join(BUFFER_DIR, "observations.jsonl");
}

function bufferObservation(obs: BufferedObservation): void {
  ensureBufferDir();
  const line = JSON.stringify(obs) + "\n";
  try {
    writeFileSync(getBufferPath(), line, { flag: "a" });
    console.error(`[pi-mem] Buffered observation for ${obs.tool_name} (worker unreachable)`);
  } catch (err) {
    console.error(`[pi-mem] Failed to buffer observation: ${err}`);
  }
}

function loadBufferedObservations(): BufferedObservation[] {
  const bufferPath = getBufferPath();
  if (!existsSync(bufferPath)) return [];
  try {
    const content = readFileSync(bufferPath, "utf-8");
    return content.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
  } catch {
    return [];
  }
}

async function flushBuffer(): Promise<number> {
  const observations = loadBufferedObservations();
  if (observations.length === 0) return 0;

  // Limit flush to MAX_BUFFER_SIZE to avoid overwhelming the worker
  const toFlush = observations.slice(0, MAX_BUFFER_SIZE);
  let flushed = 0;

  for (const obs of toFlush) {
    const { buffered_at, ...payload } = obs;
    const result = await workerPost("/api/sessions/observations", payload);
    if (result !== null) {
      flushed++;
    } else {
      // Worker went down again during flush — stop and keep remaining
      break;
    }
  }

  if (flushed > 0) {
    console.error(`[pi-mem] Flushed ${flushed}/${observations.length} buffered observations`);
    // Remove flushed observations, keep unflushed
    const remaining = observations.slice(flushed);
    ensureBufferDir();
    writeFileSync(
      getBufferPath(),
      remaining.length > 0 ? remaining.map((o) => JSON.stringify(o)).join("\n") + "\n" : ""
    );
  }

  return flushed;
}

// =============================================================================
// Project Name Derivation
// =============================================================================

function deriveProjectName(cwd: string): string {
  if (process.env.PI_MEM_PROJECT) {
    return process.env.PI_MEM_PROJECT;
  }
  const dir = basename(cwd);
  return `pi-${dir}`;
}

// =============================================================================
// Extension Factory
// =============================================================================

export default function piMemExtension(pi: ExtensionAPI) {
  // --- Extension state ---
  let contentSessionId: string | null = null;
  let projectName = "pi-agent";
  let sessionCwd = process.cwd();

  // Check kill switch
  if (process.env.PI_MEM_DISABLED === "1") {
    return;
  }

  // =========================================================================
  // Event: session_start
  //
  // Initialize local state. Attempt worker auto-start if unreachable.
  // Flush any buffered observations from previous sessions.
  // =========================================================================

  pi.on("session_start", async (_event, ctx) => {
    sessionCwd = ctx.cwd;
    projectName = deriveProjectName(sessionCwd);
    contentSessionId = `pi-${projectName}-${Date.now()}`;

    // Persist session ID into the session file for compaction recovery
    pi.appendEntry("pi-mem-session", { contentSessionId, projectName });

    // Gap 2: Check worker health, auto-start if needed
    const reachable = await isWorkerReachable();
    if (!reachable) {
      workerHealthy = false;
      console.error("[pi-mem] Worker unreachable, attempting auto-start...");
      await attemptWorkerStart();
    } else {
      workerHealthy = true;
    }

    // Gap 1: Flush any buffered observations from previous sessions
    if (workerHealthy) {
      await flushBuffer();
    }

    // Dead-man's switch: warn if observation count hasn't increased since last session
    if (workerHealthy) {
      try {
        const statsText = await workerGetText(`/api/stats`);
        if (statsText) {
          const stats = JSON.parse(statsText);
          const currentCount = stats?.database?.observations;
          if (typeof currentCount === 'number') {
            const lastCountFile = join(homedir(), '.claude-mem', 'pi-mem-last-count');
            try {
              const lastCount = parseInt(readFileSync(lastCountFile, 'utf-8').trim(), 10);
              if (!isNaN(lastCount) && currentCount <= lastCount) {
                ctx.ui.notify(`⚠️ pi-mem: Observation count unchanged (${currentCount}) — worker may be silently failing`, 'warning');
              }
            } catch { /* first run or file missing — no comparison possible */ }
            try { writeFileSync(lastCountFile, String(currentCount)); } catch { /* ignore */ }
          }
        }
      } catch { /* stats endpoint unavailable — skip check */ }
    }
  });

  // =========================================================================
  // Gap 7: Periodic health monitoring
  //
  // Check worker health every 30s and attempt restart if down.
  // Also flush buffer on recovery.
  // =========================================================================

  setInterval(async () => {
    const wasDown = !workerHealthy;
    workerHealthy = await isWorkerReachable();

    if (!workerHealthy) {
      // Attempt restart every cycle when down
      workerHealthy = await attemptWorkerStart();
    }

    if (wasDown && workerHealthy) {
      console.error("[pi-mem] Worker recovered, flushing buffer...");
      await flushBuffer();
    }
  }, HEALTH_CHECK_INTERVAL_MS);

  // =========================================================================
  // Event: before_agent_start
  //
  // Gap 6: Pass user prompt to worker for search indexing.
  // =========================================================================

  pi.on("before_agent_start", async (event) => {
    if (!contentSessionId) return;

    // Gap 2: If worker was down, try again before session init
    if (!workerHealthy) {
      workerHealthy = await isWorkerReachable();
      if (!workerHealthy) {
        workerHealthy = await attemptWorkerStart();
      }
      if (workerHealthy) {
        await flushBuffer();
      }
    }

    await workerPost("/api/sessions/init", {
      contentSessionId,
      project: projectName,
      prompt: event.prompt || "pi-agent session",
      platformSource: PLATFORM_SOURCE,
    });

    return undefined;
  });

  // =========================================================================
  // Event: context
  //
  // Gap 3: Use configurable observation count for context injection.
  // Gap 5: Pass search filters to context endpoint.
  // =========================================================================

  pi.on("context", async (event) => {
    if (!contentSessionId) return;

    const projects = encodeURIComponent(projectName);
    const contextText = await workerGetText(
      `/api/context/inject?projects=${projects}&observations=${CONTEXT_OBSERVATION_COUNT}`
    );

    if (!contextText || contextText.trim().length === 0) return;

    return {
      messages: [
        ...event.messages,
        {
          role: "user" as const,
          content: [
            {
              type: "text" as const,
              text: `<pi-mem-context>\n${contextText}\n</pi-mem-context>`,
            },
          ],
        },
      ],
    };
  });

  // =========================================================================
  // Event: tool_result
  //
  // Gap 1: Buffer observations locally if worker is unreachable.
  // =========================================================================

  pi.on("tool_result", (event) => {
    if (!contentSessionId) return;

    const toolName = event.toolName;
    if (!toolName) return;

    // Skip memory tools to prevent recursive observation loops
    if (toolName === "memory_recall" || toolName === "get_observations") return;

    // Mode system: skip tools not in active mode
    if (activeMode.captureTools.length > 0 && !activeMode.captureTools.includes(toolName)) return;
    if (activeMode.skipTools.includes(toolName)) return;

    // Dedup: per-tool cooldown — suppress rapid-fire observations from same tool within window
    if (DEDUP_WINDOW_MS > 0) {
      const dedupKey = `${contentSessionId}:${toolName}`;
      const lastSent = recentToolSends.get(dedupKey);
      if (lastSent && Date.now() - lastSent < DEDUP_WINDOW_MS) return;
      recentToolSends.set(dedupKey, Date.now());
    }

    // Extract result text from content blocks
    let toolResponseText = "";
    if (Array.isArray(event.content)) {
      toolResponseText = event.content
        .filter((block): block is { type: "text"; text: string } => block.type === "text" && "text" in block)
        .map((block) => block.text)
        .join("\n");
    }

    // Truncate to prevent oversized payloads
    if (toolResponseText.length > MAX_TOOL_RESPONSE_LENGTH) {
      toolResponseText = toolResponseText.slice(0, MAX_TOOL_RESPONSE_LENGTH - 12) + " [truncated]";
    }

    const observation = {
      contentSessionId,
      tool_name: toolName,
      tool_input: (event.input || {}) as Record<string, unknown>,
      tool_response: toolResponseText,
      cwd: sessionCwd,
      platformSource: PLATFORM_SOURCE,
    };

    if (workerHealthy) {
      workerPostFireAndForget("/api/sessions/observations", observation);
    } else {
      // Gap 1: Buffer locally when worker is down
      bufferObservation({ ...observation, buffered_at: new Date().toISOString() });
    }

    return undefined;
  });

  // =========================================================================
  // Event: agent_end
  // =========================================================================

  pi.on("agent_end", async (event) => {
    if (!contentSessionId) return;

    // Extract last assistant message for summarization
    let lastAssistantMessage = "";
    if (Array.isArray(event.messages)) {
      for (let i = event.messages.length - 1; i >= 0; i--) {
        const msg = event.messages[i];
        if (msg?.role === "assistant") {
          if (typeof msg.content === "string") {
            lastAssistantMessage = msg.content;
          } else if (Array.isArray(msg.content)) {
            lastAssistantMessage = msg.content
              .filter((block): block is { type: "text"; text: string } => block.type === "text" && "text" in block)
              .map((block) => block.text)
              .join("\n");
          }
          break;
        }
      }
    }

    await workerPost("/api/sessions/summarize", {
      contentSessionId,
      last_assistant_message: lastAssistantMessage,
      platformSource: PLATFORM_SOURCE,
    });

    const sid = contentSessionId;
    setTimeout(() => {
      workerPostFireAndForget("/api/sessions/complete", {
        contentSessionId: sid,
        platformSource: PLATFORM_SOURCE,
      });
    }, SESSION_COMPLETE_DELAY_MS);
  });

  // =========================================================================
  // Event: session_compact
  // =========================================================================

  pi.on("session_compact", () => {
    // contentSessionId persists in extension state.
  });

  // =========================================================================
  // Event: session_shutdown
  // =========================================================================

  pi.on("session_shutdown", () => {
    // Final buffer flush attempt
    if (!workerHealthy) {
      // No await — best effort
      console.error("[pi-mem] Session shutdown with unflushed buffer, observations preserved for next session");
    }
    contentSessionId = null;
  });

  // =========================================================================
  // Tool: memory_recall
  //
  // Gap 5: Enhanced search with type filter and date range.
  // =========================================================================

  pi.registerTool({
    name: "memory_recall",
    label: "Memory Recall",
    description:
      "Search past work sessions for relevant context. Use when the user asks about previous work, or when you need context about how something was done before.",
    parameters: Type.Object({
      query: Type.String({ description: "Natural language search query" }),
      limit: Type.Optional(Type.Number({ description: "Max results to return (default: 5, max: 100)" })),
      type: Type.Optional(Type.String({
        description: "Filter by observation type: discovery, bugfix, feature, refactor, change, decision",
      })),
    }),

    async execute(_toolCallId, params) {
      const query = encodeURIComponent(String(params.query));
      const limit = Math.max(1, Math.min(typeof params.limit === "number" ? Math.floor(params.limit) : 5, MAX_SEARCH_LIMIT));
      const project = encodeURIComponent(projectName);

      let searchPath = `/api/search?query=${query}&limit=${limit}&project=${project}`;

      // Gap 5: Type filter support
      if (params.type) {
        searchPath += `&type=${encodeURIComponent(String(params.type))}`;
      }

      const result = await workerGetText(searchPath);

      const text = result || "No matching memories found.";
      return {
        content: [{ type: "text" as const, text }],
        details: undefined,
      };
    },
  });

  // =========================================================================
  // Tool: get_observations — Progressive Disclosure
  //
  // Fetches full details for specific observation IDs shown in the context index.
  // This is the second half of progressive disclosure: index shows summaries,
  // this tool fetches full narratives when details are needed.
  // =========================================================================

  pi.registerTool({
    name: "get_observations",
    label: "Get Observations",
    description:
      "Fetch full details for specific observation IDs from the pi-mem context index. Use when you see observation IDs in the pi-mem-context block and need the complete narrative, facts, and files for those observations.",
    parameters: Type.Object({
      ids: Type.Array(Type.String({ description: "Observation IDs (e.g. ['8183', '8233'])" })),
    }),

    async execute(_toolCallId, params) {
      const ids = params.ids;
      if (!Array.isArray(ids) || ids.length === 0) {
        return { content: [{ type: "text" as const, text: "No IDs provided." }], details: undefined };
      }

      const idsParam = encodeURIComponent(ids.join(","));
      const result = await workerGetText(`/api/observations?ids=${idsParam}`);

      const text = result || "No observations found for given IDs.";
      return {
        content: [{ type: "text" as const, text }],
        details: undefined,
      };
    },
  });

  // =========================================================================
  // Tool: corpus_manage — Corpus Management
  //
  // Build, list, query, and delete knowledge corpora from the worker.
  // Corpora are curated collections of observations filtered by type/project.
  // =========================================================================

  pi.registerTool({
    name: "corpus_manage",
    label: "Corpus Management",
    description:
      "Manage knowledge corpora — curated collections of observations. Actions: list (show all corpora), build (create corpus from filtered observations), get (show corpus details), query (search within corpus), delete (remove corpus).",
    parameters: Type.Object({
      action: Type.String({
        description: "Action: list, build, get, query, delete",
        enum: ["list", "build", "get", "query", "delete"],
      }),
      name: Type.Optional(Type.String({ description: "Corpus name (for build, get, query, delete)" })),
      types: Type.Optional(Type.Array(Type.String({ description: "Filter by observation types" }))),
      projects: Type.Optional(Type.Array(Type.String({ description: "Filter by project names" }))),
      query: Type.Optional(Type.String({ description: "Search query (for query action)" })),
    }),

    async execute(_toolCallId, params) {
      const action = String(params.action);

      if (action === "list") {
        const result = await workerGetText("/api/corpus");
        return {
          content: [{ type: "text" as const, text: result || "No corpora found." }],
          details: undefined,
        };
      }

      if (action === "build") {
        const body: Record<string, unknown> = {
          name: params.name || `corpus-${Date.now()}`,
        };
        if (Array.isArray(params.types)) body.types = params.types;
        if (Array.isArray(params.projects)) body.projects = params.projects;

        const result = await workerPost("/api/corpus", body);
        return {
          content: [{ type: "text" as const, text: result ? JSON.stringify(result, null, 2) : "Failed to build corpus." }],
          details: undefined,
        };
      }

      if (action === "get") {
        const corpusName = encodeURIComponent(String(params.name));
        const result = await workerGetText(`/api/corpus/${corpusName}`);
        return {
          content: [{ type: "text" as const, text: result || "Corpus not found." }],
          details: undefined,
        };
      }

      if (action === "query") {
        const corpusName = encodeURIComponent(String(params.name));
        const result = await workerPost(`/api/corpus/${corpusName}/query`, {
          query: params.query || "",
        });
        return {
          content: [{ type: "text" as const, text: result ? JSON.stringify(result, null, 2) : "Query returned no results." }],
          details: undefined,
        };
      }

      if (action === "delete") {
        const corpusName = encodeURIComponent(String(params.name));
        const result = await workerPost(`/api/corpus/${corpusName}`, { _method: "DELETE" });
        return {
          content: [{ type: "text" as const, text: result ? JSON.stringify(result, null, 2) : "Failed to delete corpus." }],
          details: undefined,
        };
      }

      return {
        content: [{ type: "text" as const, text: `Unknown action: ${action}. Use: list, build, get, query, delete` }],
        details: undefined,
      };
    },
  });

  // =========================================================================
  // Command: /memory-status
  //
  // Enhanced with buffer status and worker health info.
  // =========================================================================

  pi.registerCommand("memory-status", {
    description: "Show pi-mem connection status, worker health, and buffer state",
    handler: async (_args, ctx) => {
      const reachable = await isWorkerReachable();
      const bufferCount = loadBufferedObservations().length;

      if (reachable) {
        try {
          const { controller, clear } = createTimeoutController();
          const response = await fetch(workerUrl("/api/health"), { signal: controller.signal });
          clear();
          if (response.ok) {
            const data = (await response.json()) as Record<string, unknown>;
            ctx.ui.notify(
              `pi-mem: ✅ connected to worker v${data.version || "?"} | session: ${contentSessionId || "none"} | project: ${projectName} | buffer: ${bufferCount} queued`,
              "info",
            );
          } else {
            ctx.ui.notify(`pi-mem: ⚠️ worker returned HTTP ${response.status} | buffer: ${bufferCount} queued`, "warning");
          }
        } catch {
          ctx.ui.notify(`pi-mem: ❌ worker not reachable | buffer: ${bufferCount} queued`, "error");
        }
      } else {
        ctx.ui.notify(
          `pi-mem: ❌ worker DOWN at ${workerUrl("/api/health")} | buffer: ${bufferCount} queued | auto-start attempted`,
          "error",
        );
      }
    },
  });

  // =========================================================================
  // Gap 4: Command: /memory-viewer
  //
  // Opens the claude-mem viewer UI in the browser.
  // =========================================================================

  pi.registerCommand("memory-viewer", {
    description: "Open the claude-mem viewer UI in your browser",
    handler: async (_args, ctx) => {
      const viewerUrl = `http://${WORKER_HOST}:${WORKER_PORT}`;
      const reachable = await isWorkerReachable();

      if (!reachable) {
        ctx.ui.notify("pi-mem: Worker not reachable — cannot open viewer", "error");
        return;
      }

      // Try to open browser
      const { exec } = await import("node:child_process");
      const platform = process.platform;
      const cmd = platform === "darwin" ? "open" : platform === "win32" ? "start" : "xdg-open";

      exec(`${cmd} ${viewerUrl}`, (err) => {
        if (err) {
          ctx.ui.notify(`pi-mem: Viewer at ${viewerUrl} — open manually in browser`, "info");
        } else {
          ctx.ui.notify(`pi-mem: Viewer opened at ${viewerUrl}`, "info");
        }
      });
    },
  });
}
