export interface Option {
  id: string;
  label: string;
  detail: string;
  model?: string | null;
}

export interface PreparationOptions {
  tokenizers: Option[];
  chunkers: Option[];
  maximum_comparisons: number;
  maximum_document_characters: number;
  default_chunk_size: number;
  default_chunk_overlap: number;
}

export interface PairSelection {
  id: string;
  tokenizer: string;
  chunker: string;
}

export interface ComparisonRequest {
  run_id: string;
  document: string;
  pairs: PairSelection[];
  chunk_size: number;
  chunk_overlap: number;
}

export interface TokenPiece {
  index: number;
  id: number;
  piece: string;
  decoded: string;
  start?: number | null;
  end?: number | null;
}

export interface TokenizationResult {
  tokenizer_id: string;
  model: string;
  token_count: number;
  decoded_text: string;
  tokens: TokenPiece[];
  duration_ms: number;
}

export interface PreparedChunk {
  index: number;
  text: string;
  decoded_text: string;
  start?: number | null;
  end?: number | null;
  token_count: number;
  tokens: TokenPiece[];
}

export interface ChunkingResult {
  chunker_id: string;
  chunk_count: number;
  chunks: PreparedChunk[];
  duration_ms: number;
  minimum_tokens: number;
  maximum_tokens: number;
  average_tokens: number;
}

export type ComparisonEvent =
  | { type: "run.started"; run_id: string; pair_count: number }
  | {
      type: "pair.tokenized";
      run_id: string;
      pair_id: string;
      result: TokenizationResult;
    }
  | {
      type: "pair.chunked";
      run_id: string;
      pair_id: string;
      result: ChunkingResult;
    }
  | { type: "pair.failed"; run_id: string; pair_id: string; message: string }
  | { type: "run.completed"; run_id: string; succeeded: number; failed: number }
  | { type: "run.failed"; run_id?: string | null; message: string };

export type PairStatus = "idle" | "tokenizing" | "chunking" | "complete" | "failed";

export interface PairResultState {
  status: PairStatus;
  tokenization?: TokenizationResult;
  chunking?: ChunkingResult;
  error?: string;
}
