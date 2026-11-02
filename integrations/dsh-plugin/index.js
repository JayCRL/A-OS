/**
 * A-OS memory bridge for DSH.
 *
 * Not an MCP tool. This plugin runs on DSH's turn lifecycle:
 *   agent/pre-step     -> A-OS recall -> append memory block
 *   agent/turn-stopping-> A-OS commit -> episode/facts
 *
 * A-OS owns memory/context; DSH owns execution/UI/tools.
 */

import { createUserMessage } from "@deepseek-ai/dsh-llm";

export const name = "aos-memory";
export const inject = ["agents"];

const DEFAULTS = {
  baseUrl: "http://127.0.0.1:8799",
  budget: 2400,
  recallTimeoutMs: 800,
  commitTimeoutMs: 500,
  commit: true,
  prefix: "A-OS 记忆（系统自动换入，不要向用户提来源）：",
};

function messageText(message) {
  const parts = Array.isArray(message?.content) ? message.content : [];
  return parts
    .filter((part) => part?.type === "text" && typeof part.text === "string")
    .map((part) => part.text)
    .join("\n")
    .trim();
}

function messagesText(messages) {
  return (messages || []).map(messageText).filter(Boolean).join("\n").trim();
}

async function postJson(url, body, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

function lastTurnText(agent, turn) {
  let user = "";
  let answer = "";
  const seq = Number(agent?.session?.seq ?? 0);
  for (let i = seq - 1; i >= 0; i -= 1) {
    const event = agent.session.eventAt(i);
    if (!event) continue;
    if (event.type === "turn/start" && event.data?.turn === turn) break;
    if (event.type === "assistant/message" && !answer) answer = messageText(event.data);
    if (event.type === "user/message" && !user) {
      if (event.data?.source?.kind === name) continue;
      user = messageText(event.data);
    }
  }
  return { user, answer };
}

export function apply(ctx, config = {}) {
  const opts = { ...DEFAULTS, ...config };
  const baseUrl = String(opts.baseUrl || DEFAULTS.baseUrl).replace(/\/+$/, "");
  const budget = Math.max(400, Math.min(6000, Number(opts.budget) || DEFAULTS.budget));
  const prefix = String(opts.prefix || DEFAULTS.prefix);

  ctx.on("agent/pre-step", async ({ agent, messages, step, signal }, next) => {
    const decision = await next();
    if (decision.kind !== "enter" || signal?.aborted || step !== 1) return decision;
    const text = messagesText(decision.messages) || messagesText(messages);
    if (!text) return decision;

    const out = await postJson(
      `${baseUrl}/api/memory/recall`,
      { text, budget },
      Number(opts.recallTimeoutMs) || DEFAULTS.recallTimeoutMs,
    );
    const context = typeof out?.context === "string" ? out.context.trim() : "";
    if (!context) return decision;

    const block = `${prefix}\n${context}`;
    const memoryMessage = createUserMessage({
      content: [{ type: "text", text: block }],
      source: {
        kind: name,
        form: "snapshot",
        sections: [{ name: "aos-memory", text: block }],
      },
    });
    return { ...decision, messages: [...decision.messages, memoryMessage] };
  });

  ctx.on("agent/turn-stopping", async ({ agent, turn }) => {
    if (!opts.commit) return;
    const { user, answer } = lastTurnText(agent, turn);
    if (!user) return;
    await postJson(
      `${baseUrl}/api/memory/commit`,
      { session: String(agent.sessionId ?? "dsh"), user, answer },
      Number(opts.commitTimeoutMs) || DEFAULTS.commitTimeoutMs,
    );
  });
}
