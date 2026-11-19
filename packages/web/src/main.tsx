import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import { ToastProvider } from "./components/Toast";
import { getThemeController } from "./theme/theme";
import "./styles/tokens.css";
import "./styles/app.css";

// The inline script in index.html already set data-theme for the first paint;
// the controller keeps system preference and later changes in sync.
getThemeController();

const container = document.getElementById("root");
if (!container) throw new Error("SuperTuner: #root element missing from index.html");

createRoot(container).render(
  <StrictMode>
    <BrowserRouter>
      <ToastProvider>
        <App />
      </ToastProvider>
    </BrowserRouter>
  </StrictMode>,
);