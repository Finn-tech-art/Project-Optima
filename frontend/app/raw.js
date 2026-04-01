const rawPageResponse = document.getElementById("raw-page-response");
const refreshRawButton = document.getElementById("refresh-raw");
const STORAGE_KEY = "project_optima_latest_run";
const THEME_KEY = "project_optima_theme";
const themeToggle = document.getElementById("theme-toggle");

// This block applies the saved or requested visual theme to the raw-data page.
// It takes: a theme string such as 'light' or 'dark'.
// It gives: synchronized DOM theme state and toggle copy on this page.
function applyTheme(theme) {
  const resolvedTheme = theme === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", resolvedTheme);
  window.localStorage.setItem(THEME_KEY, resolvedTheme);
  if (themeToggle) {
    themeToggle.textContent = resolvedTheme === "dark" ? "Light Mode" : "Dark Mode";
  }
}

// This block restores the persisted theme preference when the raw-data page opens.
// It takes: the browser's localStorage state.
// It gives: the same visual theme as the main dashboard.
function initializeTheme() {
  const storedTheme = window.localStorage.getItem(THEME_KEY) || "light";
  applyTheme(storedTheme);
}

// This block renders the latest cached graph payload on the dedicated raw-data page.
// It takes: the current browser localStorage contents.
// It gives: a readable JSON snapshot or a fallback message when no run is cached yet.
function renderRawSnapshot() {
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (!stored) {
    rawPageResponse.textContent = "No run payload has been stored yet.";
    return;
  }

  try {
    const parsed = JSON.parse(stored);
    rawPageResponse.textContent = JSON.stringify(parsed, null, 2);
  } catch (error) {
    rawPageResponse.textContent = "Stored payload could not be parsed.";
  }
}

refreshRawButton.addEventListener("click", renderRawSnapshot);
themeToggle.addEventListener("click", () => {
  const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
  applyTheme(currentTheme === "dark" ? "light" : "dark");
});
initializeTheme();
renderRawSnapshot();
