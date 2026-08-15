import { For, Show } from "solid-js";

import type { Option, PairResultState, PairSelection } from "../types";
import { TokenStream } from "./TokenStream";

interface ComparisonCardProps {
  index: number;
  pair: PairSelection;
  state: PairResultState;
  tokenizers: Option[];
  chunkers: Option[];
  removable: boolean;
  onChange: (pair: PairSelection) => void;
  onRemove: () => void;
}

const statusLabels: Record<PairResultState["status"], string> = {
  idle: "Ready",
  tokenizing: "Tokenizing…",
  chunking: "Chunking…",
  complete: "Complete",
  failed: "Failed",
};

export function ComparisonCard(props: ComparisonCardProps) {
  const tokenizer = () => props.tokenizers.find((item) => item.id === props.pair.tokenizer);
  const chunker = () => props.chunkers.find((item) => item.id === props.pair.chunker);

  return (
    <article class="comparison-card" aria-label={`Comparison ${props.index + 1}`}>
      <header class="card-header">
        <div>
          <span class="eyebrow">Pair {props.index + 1}</span>
          <h2>{tokenizer()?.label ?? props.pair.tokenizer}</h2>
          <p>with {chunker()?.label ?? props.pair.chunker}</p>
        </div>
        <div class="card-actions">
          <span class={`status status-${props.state.status}`} aria-live="polite">
            {statusLabels[props.state.status]}
          </span>
          <Show when={props.removable}>
            <button class="icon-button" type="button" onClick={props.onRemove} aria-label={`Remove pair ${props.index + 1}`}>
              ×
            </button>
          </Show>
        </div>
      </header>

      <div class="pair-selectors">
        <label>
          <span>Tokenizer</span>
          <select
            aria-label={`Tokenizer for pair ${props.index + 1}`}
            value={props.pair.tokenizer}
            onChange={(event) => props.onChange({ ...props.pair, tokenizer: event.currentTarget.value })}
          >
            <For each={props.tokenizers}>{(item) => <option value={item.id}>{item.label}</option>}</For>
          </select>
        </label>
        <label>
          <span>Chunker</span>
          <select
            aria-label={`Chunker for pair ${props.index + 1}`}
            value={props.pair.chunker}
            onChange={(event) => props.onChange({ ...props.pair, chunker: event.currentTarget.value })}
          >
            <For each={props.chunkers}>{(item) => <option value={item.id}>{item.label}</option>}</For>
          </select>
        </label>
      </div>

      <div class="model-note">
        <span>{tokenizer()?.model}</span>
        <span>{chunker()?.detail}</span>
      </div>

      <Show when={props.state.status === "idle"}>
        <div class="empty-card-state">
          <span aria-hidden="true">Aa → ▦</span>
          <p>Run the matrix to reveal tokens, then chunk boundaries.</p>
        </div>
      </Show>

      <Show when={props.state.status === "tokenizing"}>
        <div class="processing-state" aria-label="Tokenization in progress">
          <span /><span /><span />
          <p>Reading the document vocabulary…</p>
        </div>
      </Show>

      <Show when={props.state.tokenization} keyed>
        {(result) => (
          <section class="result-stage token-stage">
            <div class="stage-heading">
              <div>
                <span class="stage-number">01</span>
                <h3>Tokenized</h3>
              </div>
              <dl class="mini-metrics">
                <div><dt>Tokens</dt><dd>{result.token_count.toLocaleString()}</dd></div>
                <div><dt>Time</dt><dd>{result.duration_ms.toFixed(1)} ms</dd></div>
              </dl>
            </div>
            <TokenStream tokens={result.tokens} />
            <details class="raw-details">
              <summary>Show raw token IDs</summary>
              <code>{result.tokens.map((token) => token.id).join(", ")}</code>
            </details>
          </section>
        )}
      </Show>

      <Show when={props.state.status === "chunking"}>
        <div class="chunking-divider" aria-live="polite">
          <span class="line" />
          <span>Finding boundaries…</span>
          <span class="line" />
        </div>
      </Show>

      <Show when={props.state.chunking} keyed>
        {(result) => (
          <section class="result-stage chunk-stage">
            <div class="stage-heading">
              <div>
                <span class="stage-number">02</span>
                <h3>Chunked</h3>
              </div>
              <dl class="mini-metrics">
                <div><dt>Chunks</dt><dd>{result.chunk_count}</dd></div>
                <div><dt>Average</dt><dd>{result.average_tokens} tok</dd></div>
              </dl>
            </div>
            <div class="chunk-list">
              <For each={result.chunks}>
                {(chunk) => (
                  <article class="chunk" style={{ "--chunk-index": chunk.index }}>
                    <header>
                      <strong>Chunk {chunk.index + 1}</strong>
                      <span>{chunk.token_count} tokens</span>
                    </header>
                    <p class="decoded-chunk">{chunk.decoded_text}</p>
                    <TokenStream tokens={chunk.tokens} compact />
                  </article>
                )}
              </For>
            </div>
          </section>
        )}
      </Show>

      <Show when={props.state.error} keyed>
        {(message) => <p class="pair-error" role="alert">{message}</p>}
      </Show>
    </article>
  );
}
