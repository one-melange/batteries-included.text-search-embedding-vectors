import type { ComparisonEvent, ComparisonRequest } from "../types";

export interface ComparisonSocket {
  send(request: ComparisonRequest): void;
  close(): void;
}

export type SocketFactory = (
  onEvent: (event: ComparisonEvent) => void,
  onConnectionError: (message: string) => void,
) => ComparisonSocket;

function websocketUrl(): string {
  const configured = import.meta.env.VITE_DOCUMENT_PREPARATION_WS_URL;
  if (configured) return configured;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = import.meta.env.DEV
    ? `${window.location.hostname}:8000`
    : window.location.host;
  return `${protocol}//${host}/ws/document-preparation`;
}

export const createComparisonSocket: SocketFactory = (onEvent, onConnectionError) => {
  let socket: WebSocket | undefined;
  let queued: ComparisonRequest | undefined;
  let closed = false;

  const connect = () => {
    if (closed || socket?.readyState === WebSocket.OPEN || socket?.readyState === WebSocket.CONNECTING) {
      return;
    }

    socket = new WebSocket(websocketUrl());
    socket.addEventListener("open", () => {
      if (queued) {
        socket?.send(JSON.stringify(queued));
        queued = undefined;
      }
    });
    socket.addEventListener("message", (message) => {
      try {
        onEvent(JSON.parse(String(message.data)) as ComparisonEvent);
      } catch {
        onConnectionError("The server returned an unreadable update.");
      }
    });
    socket.addEventListener("error", () => {
      onConnectionError("The comparison service is unavailable.");
    });
    socket.addEventListener("close", () => {
      socket = undefined;
    });
  };

  connect();

  return {
    send(request) {
      queued = request;
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(request));
        queued = undefined;
      } else {
        connect();
      }
    },
    close() {
      closed = true;
      queued = undefined;
      socket?.close();
    },
  };
};
