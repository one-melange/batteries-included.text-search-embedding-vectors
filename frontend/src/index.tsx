import { Route, Router } from "@solidjs/router";
import { render } from "solid-js/web";

import { AppShell } from "./App";
import { DocumentPreparation } from "./pages/DocumentPreparation";
import "./styles.css";

const root = document.getElementById("root");

if (!root) throw new Error("Missing #root element");

render(
  () => (
    <Router root={AppShell}>
      <Route
        path={["/", "/document-preparation"]}
        component={() => <DocumentPreparation />}
      />
    </Router>
  ),
  root,
);
