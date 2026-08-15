import { afterEach, describe, expect, it, vi } from "vitest";

import type { ComparisonRequest } from "../types";
import { createComparisonSocket } from "./comparisonSocket";

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readyState = FakeWebSocket.CONNECTING;
  listeners: Record<string, ((event: { data?: string }) => void)[]> = {};
  send = vi.fn();
  close = vi.fn(() => {
    this.readyState = FakeWebSocket.CLOSED;
  });

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }

  addEventListener(name: string, listener: (event: { data?: string }) => void) {
    this.listeners[name] ??= [];
    this.listeners[name].push(listener);
  }

  trigger(name: string, data?: string) {
    if (name === "open") this.readyState = FakeWebSocket.OPEN;
    this.listeners[name]?.forEach((listener) => listener({ data }));
  }
}

const request: ComparisonRequest = {
  run_id: "run-1",
  document: "hello",
  pairs: [{ id: "pair-1", tokenizer: "tiktoken", chunker: "langchain" }],
  chunk_size: 32,
  chunk_overlap: 4,
};

describe("createComparisonSocket", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    FakeWebSocket.instances = [];
  });

  it("queues the latest request until the connection opens", () => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const client = createComparisonSocket(vi.fn(), vi.fn());
    const instance = FakeWebSocket.instances[0];

    client.send(request);
    expect(instance.send).not.toHaveBeenCalled();
    instance.trigger("open");

    expect(instance.send).toHaveBeenCalledWith(JSON.stringify(request));
    client.close();
    expect(instance.close).toHaveBeenCalled();
  });

  it("delivers parsed updates and reports invalid server messages", () => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const onEvent = vi.fn();
    const onError = vi.fn();
    createComparisonSocket(onEvent, onError);
    const instance = FakeWebSocket.instances[0];

    instance.trigger("message", JSON.stringify({ type: "run.started", run_id: "run-1", pair_count: 1 }));
    instance.trigger("message", "not-json");
    instance.trigger("error");

    expect(onEvent).toHaveBeenCalledWith(expect.objectContaining({ type: "run.started" }));
    expect(onError).toHaveBeenCalledWith("The server returned an unreadable update.");
    expect(onError).toHaveBeenCalledWith("The comparison service is unavailable.");
  });
});
