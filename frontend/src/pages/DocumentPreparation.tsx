import { For, Show, createMemo, createSignal, onCleanup, onMount } from "solid-js";

import { ComparisonCard } from "../components/ComparisonCard";
import { createComparisonSocket, type ComparisonSocket, type SocketFactory } from "../lib/comparisonSocket";
import type {
  ComparisonEvent,
  ComparisonRequest,
  PairResultState,
  PairSelection,
  PreparationOptions,
} from "../types";

export const DEFAULT_OPTIONS: PreparationOptions = {
  tokenizers: [
    { id: "tiktoken", label: "tiktoken", detail: "OpenAI byte-pair encoding", model: "cl100k_base" },
    { id: "huggingface", label: "HF tokenizers", detail: "Hugging Face Rust tokenizer", model: "bert-base-uncased" },
    { id: "quicktok", label: "quicktok", detail: "Native C++ byte-pair encoding", model: "cl100k_base" },
    { id: "sentencepiece", label: "SentencePiece", detail: "Google subword tokenizer", model: "google-t5/t5-small" },
  ],
  chunkers: [
    { id: "langchain", label: "LangChain", detail: "RecursiveCharacterTextSplitter" },
    { id: "semchunk", label: "semchunk", detail: "Semantic heuristic chunking" },
    { id: "chonkie", label: "Chonkie", detail: "SemanticChunker" },
    { id: "llamaindex", label: "LlamaIndex", detail: "SentenceSplitter node parser" },
  ],
  maximum_comparisons: 4,
  maximum_document_characters: 100_000,
  default_chunk_size: 256,
  default_chunk_overlap: 32,
};

const SAMPLE_DOCUMENT = `Vector search starts before a single embedding is created. The way a document is tokenized changes its measured length, while the chunking strategy decides which ideas stay together.

A recursive splitter follows structural boundaries. A semantic splitter watches for changes in meaning. Comparing both stages makes invisible preparation choices tangible—and much easier to tune.`;

interface DocumentPreparationProps {
  socketFactory?: SocketFactory;
  loadOptions?: () => Promise<PreparationOptions>;
}

function pairKey(pair: Pick<PairSelection, "tokenizer" | "chunker">): string {
  return `${pair.tokenizer}:${pair.chunker}`;
}

function nextPair(options: PreparationOptions, pairs: PairSelection[]): PairSelection {
  const used = new Set(pairs.map(pairKey));
  for (const tokenizer of options.tokenizers) {
    for (const chunker of options.chunkers) {
      if (!used.has(`${tokenizer.id}:${chunker.id}`)) {
        return {
          id: `pair-${Date.now()}-${pairs.length}`,
          tokenizer: tokenizer.id,
          chunker: chunker.id,
        };
      }
    }
  }
  throw new Error("Every comparison pair is already selected.");
}

function runId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `run-${Date.now()}`;
}

async function fetchOptions(): Promise<PreparationOptions> {
  const response = await fetch("/api/document-preparation/options");
  if (!response.ok) throw new Error("Could not load comparison options.");
  return response.json() as Promise<PreparationOptions>;
}

export function DocumentPreparation(props: DocumentPreparationProps) {
  const [options, setOptions] = createSignal(DEFAULT_OPTIONS);
  const [document, setDocument] = createSignal(SAMPLE_DOCUMENT);
  const [pairs, setPairs] = createSignal<PairSelection[]>([
    { id: "pair-1", tokenizer: "tiktoken", chunker: "langchain" },
  ]);
  const [results, setResults] = createSignal<Record<string, PairResultState>>({
    "pair-1": { status: "idle" },
  });
  const [chunkSize, setChunkSize] = createSignal(DEFAULT_OPTIONS.default_chunk_size);
  const [chunkOverlap, setChunkOverlap] = createSignal(DEFAULT_OPTIONS.default_chunk_overlap);
  const [activeRun, setActiveRun] = createSignal<string>();
  const [notice, setNotice] = createSignal<string>();
  let socket: ComparisonSocket | undefined;

  const canAdd = createMemo(() => pairs().length < options().maximum_comparisons);
  const characterCount = createMemo(() => document().length);

  const handleEvent = (event: ComparisonEvent) => {
    if (event.run_id && activeRun() && event.run_id !== activeRun()) return;

    if (event.type === "pair.tokenized") {
      setResults((current) => ({
        ...current,
        [event.pair_id]: {
          ...current[event.pair_id],
          status: "chunking",
          tokenization: event.result,
        },
      }));
    } else if (event.type === "pair.chunked") {
      setResults((current) => ({
        ...current,
        [event.pair_id]: {
          ...current[event.pair_id],
          status: "complete",
          chunking: event.result,
        },
      }));
    } else if (event.type === "pair.failed") {
      setResults((current) => ({
        ...current,
        [event.pair_id]: { ...current[event.pair_id], status: "failed", error: event.message },
      }));
    } else if (event.type === "run.completed") {
      setNotice(event.failed ? `${event.succeeded} comparisons completed; ${event.failed} failed.` : `${event.succeeded} comparisons complete.`);
    } else if (event.type === "run.failed") {
      setNotice(event.message);
    }
  };

  onMount(() => {
    socket = (props.socketFactory ?? createComparisonSocket)(handleEvent, setNotice);
    (props.loadOptions ?? fetchOptions)()
      .then((loaded) => {
        setOptions(loaded);
        setChunkSize(loaded.default_chunk_size);
        setChunkOverlap(loaded.default_chunk_overlap);
      })
      .catch(() => setNotice("Using built-in matrix options while the service starts."));
  });
  onCleanup(() => socket?.close());

  const updatePair = (index: number, pair: PairSelection) => {
    setPairs((current) => current.map((item, itemIndex) => (itemIndex === index ? pair : item)));
    setResults((current) => ({ ...current, [pair.id]: { status: "idle" } }));
  };

  const addPair = () => {
    if (!canAdd()) return;
    const pair = nextPair(options(), pairs());
    setPairs((current) => [...current, pair]);
    setResults((current) => ({ ...current, [pair.id]: { status: "idle" } }));
  };

  const removePair = (index: number) => {
    const removed = pairs()[index];
    setPairs((current) => current.filter((_, itemIndex) => itemIndex !== index));
    setResults((current) => {
      const next = { ...current };
      delete next[removed.id];
      return next;
    });
  };

  const compare = () => {
    setNotice(undefined);
    const trimmed = document().trim();
    if (!trimmed) {
      setNotice("Add some document text before running the comparison.");
      return;
    }
    if (document().length > options().maximum_document_characters) {
      setNotice(`Documents are limited to ${options().maximum_document_characters.toLocaleString()} characters.`);
      return;
    }
    if (chunkOverlap() >= chunkSize()) {
      setNotice("Overlap must be smaller than chunk size.");
      return;
    }
    const combinations = pairs().map(pairKey);
    if (new Set(combinations).size !== combinations.length) {
      setNotice("Choose a unique tokenizer and chunker for each comparison.");
      return;
    }

    const id = runId();
    setActiveRun(id);
    setResults(
      Object.fromEntries(pairs().map((pair) => [pair.id, { status: "tokenizing" } satisfies PairResultState])),
    );
    const request: ComparisonRequest = {
      run_id: id,
      document: document(),
      pairs: pairs(),
      chunk_size: chunkSize(),
      chunk_overlap: chunkOverlap(),
    };
    socket?.send(request);
  };

  const loadFile = async (event: Event) => {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const text = await file.text();
    if (text.length > options().maximum_document_characters) {
      setNotice(`That file exceeds the ${options().maximum_document_characters.toLocaleString()} character limit.`);
      input.value = "";
      return;
    }
    setDocument(text);
    setNotice(`Loaded ${file.name}`);
  };

  return (
    <main>
      <section class="hero">
        <div>
          <span class="kicker">Preparation playground</span>
          <h1>See what your vectors will inherit.</h1>
          <p>
            Compare token boundaries first, then watch four chunking strategies reshape the same document—side by side, as each result arrives.
          </p>
        </div>
        <div class="hero-stat" aria-hidden="true">
          <strong>16</strong>
          <span>possible pairings</span>
          <div class="matrix-dots">{Array.from({ length: 16 }, () => <i />)}</div>
        </div>
      </section>

      <section class="workbench" aria-labelledby="document-heading">
        <div class="document-panel">
          <div class="section-title">
            <div>
              <span class="step">Step 1</span>
              <h2 id="document-heading">Your document</h2>
            </div>
            <label class="file-button">
              Load .txt or .md
              <input type="file" accept=".txt,.md,text/plain,text/markdown" onChange={loadFile} />
            </label>
          </div>
          <textarea
            aria-label="Document text"
            value={document()}
            maxlength={options().maximum_document_characters}
            onInput={(event) => setDocument(event.currentTarget.value)}
          />
          <div class="document-meta">
            <span>{characterCount().toLocaleString()} characters</span>
            <button type="button" onClick={() => setDocument("")}>Clear</button>
          </div>
        </div>

        <aside class="settings-panel" aria-labelledby="settings-heading">
          <div class="section-title">
            <div>
              <span class="step">Step 2</span>
              <h2 id="settings-heading">Shared controls</h2>
            </div>
          </div>
          <label>
            <span>Chunk size <small>tokens</small></span>
            <input
              aria-label="Chunk size"
              type="number"
              min="8"
              max="4096"
              value={chunkSize()}
              onInput={(event) => setChunkSize(event.currentTarget.valueAsNumber)}
            />
          </label>
          <label>
            <span>Overlap <small>tokens</small></span>
            <input
              aria-label="Chunk overlap"
              type="number"
              min="0"
              max="2048"
              value={chunkOverlap()}
              onInput={(event) => setChunkOverlap(event.currentTarget.valueAsNumber)}
            />
          </label>
          <p>One budget keeps the comparison honest across every pair.</p>
          <button class="primary-button" type="button" onClick={compare}>
            <span>Run comparison</span>
            <span aria-hidden="true">→</span>
          </button>
        </aside>
      </section>

      <Show when={notice()} keyed>
        {(message) => <p class="notice" role="status">{message}</p>}
      </Show>

      <section class="comparison-section" aria-labelledby="comparison-heading">
        <div class="comparison-title-row">
          <div>
            <span class="step">Step 3</span>
            <h2 id="comparison-heading">Compare the pipeline</h2>
            <p>Each card updates once for tokenization, then again for chunking.</p>
          </div>
          <button class="secondary-button" type="button" onClick={addPair} disabled={!canAdd()}>
            <span aria-hidden="true">＋</span> Add comparison
            <small>{pairs().length}/{options().maximum_comparisons}</small>
          </button>
        </div>

        <div class={`comparison-grid columns-${pairs().length}`}>
          <For each={pairs()}>
            {(pair, index) => (
              <ComparisonCard
                index={index()}
                pair={pair}
                state={results()[pair.id] ?? { status: "idle" }}
                tokenizers={options().tokenizers}
                chunkers={options().chunkers}
                removable={pairs().length > 1}
                onChange={(updated) => updatePair(index(), updated)}
                onRemove={() => removePair(index())}
              />
            )}
          </For>
        </div>
      </section>
    </main>
  );
}
