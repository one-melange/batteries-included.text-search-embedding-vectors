import { A } from "@solidjs/router";
import type { ParentComponent } from "solid-js";

export const AppShell: ParentComponent = (props) => (
  <div class="app-shell">
    <header class="topbar">
      <A class="brand" href="/document-preparation" aria-label="Vector Lab home">
        <span class="brand-mark" aria-hidden="true">
          VL
        </span>
        <span>
          <strong>Vector Lab</strong>
          <small>batteries included</small>
        </span>
      </A>
      <nav aria-label="Main navigation">
        <A href="/document-preparation" activeClass="active">
          Document Preparation
        </A>
      </nav>
      <div class="matrix-badge" aria-label="Sixteen combinations available">
        <span>4 × 4</span>
        <small>matrix</small>
      </div>
    </header>
    {props.children}
  </div>
);
