/* Close all <details> elements on page load and instant navigation.
   Prevents stale open state from leaking across pages when Material's
   instant navigation is enabled, and ensures mkdocstrings source-code
   blocks and collapsible admonitions start closed. */
function closeAllDetails() {
  document.querySelectorAll(".md-content details[open]").forEach(function (el) {
    el.removeAttribute("open");
  });
}

// Initial page load
document.addEventListener("DOMContentLoaded", closeAllDetails);

// Material instant navigation (fires on every page swap)
if (typeof document$ !== "undefined") {
  document$.subscribe(closeAllDetails);
}
