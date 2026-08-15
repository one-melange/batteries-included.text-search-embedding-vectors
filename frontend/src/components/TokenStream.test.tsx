import { render, screen } from "@solidjs/testing-library";
import { describe, expect, it } from "vitest";

import { TokenStream, visibleToken } from "./TokenStream";

describe("TokenStream", () => {
  it("makes whitespace and empty decoded pieces readable", () => {
    expect(visibleToken({ index: 0, id: 1, piece: " ", decoded: " " })).toBe("·");
    expect(visibleToken({ index: 1, id: 2, piece: "\n", decoded: "\n" })).toBe("↵");
    expect(visibleToken({ index: 2, id: 3, piece: "", decoded: "" })).toBe("∅");
  });

  it("renders decoded text while keeping raw ids in metadata", () => {
    render(() => (
      <TokenStream
        tokens={[{ index: 0, id: 101, piece: "hello", decoded: "hello" }]}
      />
    ));

    const token = screen.getByText("hello");
    expect(token).toHaveAttribute("data-token-id", "101");
    expect(token).toHaveAttribute("title", expect.stringContaining("101"));
  });
});
