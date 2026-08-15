import { render, screen, waitFor } from "@solidjs/testing-library";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ComparisonEvent } from "../types";
import { DEFAULT_OPTIONS, DocumentPreparation } from "./DocumentPreparation";

function setup() {
  let onEvent: ((event: ComparisonEvent) => void) | undefined;
  const send = vi.fn();
  const close = vi.fn();
  const socketFactory = vi.fn((handler: (event: ComparisonEvent) => void) => {
    onEvent = handler;
    return { send, close };
  });
  const view = render(() => (
    <DocumentPreparation
      socketFactory={socketFactory}
      loadOptions={() => Promise.resolve(DEFAULT_OPTIONS)}
    />
  ));
  return {
    send,
    close,
    unmount: view.unmount,
    emit(event: ComparisonEvent) {
      if (!onEvent) throw new Error("Socket handler is not mounted");
      onEvent(event);
    },
  };
}

describe("DocumentPreparation", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("renders the requested route content and supports four side-by-side pairs", async () => {
    const user = userEvent.setup();
    setup();

    expect(screen.getByRole("heading", { name: "See what your vectors will inherit." })).toBeInTheDocument();
    const add = screen.getByRole("button", { name: /Add comparison/ });
    await user.click(add);
    await user.click(add);
    await user.click(add);

    expect(screen.getByLabelText("Comparison 4")).toBeInTheDocument();
    expect(add).toBeDisabled();
  });

  it("sends document, settings, and selected pairs to the socket", async () => {
    const user = userEvent.setup();
    const { send } = setup();
    await waitFor(() => expect(screen.getByRole("button", { name: /Run comparison/ })).toBeEnabled());

    await user.clear(screen.getByLabelText("Document text"));
    await user.type(screen.getByLabelText("Document text"), "A focused test document.");
    await user.clear(screen.getByLabelText("Chunk size"));
    await user.type(screen.getByLabelText("Chunk size"), "64");
    await user.click(screen.getByRole("button", { name: /Run comparison/ }));

    expect(send).toHaveBeenCalledWith(
      expect.objectContaining({
        document: "A focused test document.",
        chunk_size: 64,
        chunk_overlap: 32,
        pairs: [expect.objectContaining({ tokenizer: "tiktoken", chunker: "langchain" })],
      }),
    );
    expect(screen.getByText("Tokenizing…")).toBeInTheDocument();
  });

  it("renders tokenization before the later decoded chunk update", async () => {
    const user = userEvent.setup();
    const connection = setup();
    await user.click(screen.getByRole("button", { name: /Run comparison/ }));
    const request = connection.send.mock.calls[0][0];

    connection.emit({
      type: "pair.tokenized",
      run_id: request.run_id,
      pair_id: "pair-1",
      result: {
        tokenizer_id: "tiktoken",
        model: "cl100k_base",
        token_count: 2,
        decoded_text: "Hello world",
        duration_ms: 1.2,
        tokens: [
          { index: 0, id: 9906, piece: "Hello", decoded: "Hello" },
          { index: 1, id: 1917, piece: " world", decoded: " world" },
        ],
      },
    });

    expect(screen.getByRole("heading", { name: "Tokenized" })).toBeInTheDocument();
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Chunking…")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Chunked" })).not.toBeInTheDocument();

    connection.emit({
      type: "pair.chunked",
      run_id: request.run_id,
      pair_id: "pair-1",
      result: {
        chunker_id: "langchain",
        chunk_count: 1,
        duration_ms: 2.3,
        minimum_tokens: 2,
        maximum_tokens: 2,
        average_tokens: 2,
        chunks: [
          {
            index: 0,
            text: "Hello world",
            decoded_text: "Hello world",
            token_count: 2,
            tokens: [
              { index: 0, id: 9906, piece: "Hello", decoded: "Hello" },
              { index: 1, id: 1917, piece: " world", decoded: " world" },
            ],
          },
        ],
      },
    });

    expect(screen.getByRole("heading", { name: "Chunked" })).toBeInTheDocument();
    expect(screen.getByText("Hello world")).toBeInTheDocument();
    expect(screen.getByText("Complete")).toBeInTheDocument();
  });

  it("rejects duplicate combinations and invalid overlap before sending", async () => {
    const user = userEvent.setup();
    const { send } = setup();
    await user.click(screen.getByRole("button", { name: /Add comparison/ }));
    await user.selectOptions(screen.getByLabelText("Chunker for pair 2"), "langchain");
    await user.click(screen.getByRole("button", { name: /Run comparison/ }));

    expect(screen.getByRole("status")).toHaveTextContent("unique tokenizer and chunker");
    expect(send).not.toHaveBeenCalled();

    await user.selectOptions(screen.getByLabelText("Chunker for pair 2"), "semchunk");
    await user.clear(screen.getByLabelText("Chunk overlap"));
    await user.type(screen.getByLabelText("Chunk overlap"), "256");
    await user.click(screen.getByRole("button", { name: /Run comparison/ }));
    expect(screen.getByRole("status")).toHaveTextContent("Overlap must be smaller");
    expect(send).not.toHaveBeenCalled();
  });

  it("closes the live connection when the page unmounts", () => {
    const connection = setup();
    connection.unmount();
    expect(connection.close).toHaveBeenCalledOnce();
  });
});
