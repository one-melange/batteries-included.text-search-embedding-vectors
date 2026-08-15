import { For, Show } from "solid-js";

import type { TokenPiece } from "../types";

interface TokenStreamProps {
  tokens: TokenPiece[];
  compact?: boolean;
}

export function visibleToken(token: TokenPiece): string {
  const value = token.decoded || token.piece;
  if (!value) return "∅";
  return value.replaceAll(" ", "·").replaceAll("\t", "⇥").replaceAll("\n", "↵");
}

export function TokenStream(props: TokenStreamProps) {
  return (
    <div classList={{ "token-stream": true, compact: Boolean(props.compact) }}>
      <For each={props.tokens}>
        {(token) => (
          <span
            class="token-chip"
            title={`Token ${token.id} · ${JSON.stringify(token.piece)}`}
            data-token-id={token.id}
          >
            {visibleToken(token)}
          </span>
        )}
      </For>
      <Show when={props.tokens.length === 0}>
        <span class="empty-value">No tokens</span>
      </Show>
    </div>
  );
}
