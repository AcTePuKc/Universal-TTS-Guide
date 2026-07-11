document$.subscribe(function () {
  if (typeof mermaid === "undefined") {
    return;
  }

  if (window.location.protocol === "file:") {
    return;
  }

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
    theme: document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "default",
    flowchart: {
      useMaxWidth: true,
      htmlLabels: true
    }
  });

  const diagrams = document.querySelectorAll("pre.mermaid, code.mermaid");
  if (diagrams.length) {
    try {
      const result = mermaid.run({
        nodes: diagrams
      });
      if (result && typeof result.catch === "function") {
        result.catch(function () {
          // Keep the rest of the documentation usable if one diagram is invalid.
        });
      }
    } catch (error) {
      // Keep the rest of the documentation usable if one diagram is invalid.
    }
  }
});
