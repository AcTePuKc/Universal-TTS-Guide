(function () {
  const languages = [
    { code: "en", segment: "", native: "English" },
    { code: "bg", segment: "bg", native: "Български" },
    { code: "es", segment: "es", native: "Español" },
    { code: "fr", segment: "fr", native: "Français" },
    { code: "it", segment: "it", native: "Italiano" }
  ];

  const guides = [
    "guides/1-data-preparation/",
    "guides/2-training-setup/",
    "guides/3-model-training/",
    "guides/4-inference/",
    "guides/5-packaging-and-sharing/",
    "guides/6-troubleshooting-and-resources/"
  ];

  const labels = {
    en: { language: "Language", toc: "Table of contents", previous: "Previous", next: "Next" },
    bg: { language: "Език", toc: "Съдържание", previous: "Предишна", next: "Следваща" },
    es: { language: "Idioma", toc: "Tabla de contenidos", previous: "Anterior", next: "Siguiente" },
    fr: { language: "Langue", toc: "Table des matières", previous: "Précédente", next: "Suivante" },
    it: { language: "Lingua", toc: "Indice", previous: "Precedente", next: "Successiva" }
  };

  const guideTitles = {
    en: ["Data Preparation", "Training Setup", "Model Training", "Inference", "Packaging and Sharing", "Troubleshooting and Resources"],
    bg: ["Подготовка на данни", "Настройка за обучение", "Обучение на модела", "Извеждане", "Пакетиране и споделяне", "Отстраняване на проблеми и ресурси"],
    es: ["Preparación de Datos", "Configuración del Entrenamiento", "Entrenamiento del Modelo", "Inferencia", "Empaquetado y Compartición", "Resolución de Problemas y Recursos"],
    fr: ["Préparation des données", "Configuration de l'entraînement", "Entraînement du modèle", "Inférence", "Packaging et partage", "Dépannage et ressources"],
    it: ["Preparazione dei Dati", "Configurazione dell'Addestramento", "Addestramento del Modello", "Inferenza", "Packaging e Condivisione", "Risoluzione dei Problemi e Risorse"]
  };

  const topNavigation = {
    en: ["Home", "Guides", "Glossary", "Translation Guide", "Licence"],
    bg: ["Начало", "Ръководства", "Речник", "Ръководство за превод", "Лиценз"],
    es: ["Inicio", "Guías", "Glosario", "Guía de Traducción", "Licencia"],
    fr: ["Accueil", "Guides", "Glossaire", "Guide de traduction", "Licence"],
    it: ["Inizio", "Guide", "Glossario", "Guida alla Traduzione", "Licenza"]
  };

  const pathname = window.location.pathname;
  const siteMarker = "/site/";
  const fileSiteIndex = pathname.toLowerCase().lastIndexOf(siteMarker);
  let base;
  let rawPath;

  if (window.location.protocol === "file:" && fileSiteIndex >= 0) {
    base = pathname.slice(0, fileSiteIndex + siteMarker.length);
    rawPath = pathname.slice(fileSiteIndex + siteMarker.length);
  } else {
    const repositoryMarker = "/Universal-TTS-Guide/";
    const repositoryIndex = pathname.indexOf(repositoryMarker);
    base = repositoryIndex >= 0 ? pathname.slice(0, repositoryIndex + repositoryMarker.length) : "/";
    rawPath = pathname.slice(base.length);
  }

  rawPath = rawPath.replace(/index\.html$/, "").replace(/^\/+|\/+$/g, "");
  const parts = rawPath.split("/").filter(Boolean);
  const language = languages.find((item) => item.segment && parts[0] === item.segment) || languages[0];
  const page = language.segment ? parts.slice(1).join("/") : parts.join("/");
  const pagePath = page ? `${page}/` : "";

  function localizedPath(target) {
    return `${base}${language.segment ? `${language.segment}/` : ""}${target}`;
  }

  function resolvedPath(target) {
    const localTarget = window.location.protocol === "file:" && target.endsWith("/")
      ? `${target}index.html`
      : target;
    return new URL(localTarget, window.location.href).href;
  }

  function addLanguageSelector() {
    const header = document.querySelector(".md-header__inner");
    if (!header || document.querySelector(".site-language-switcher")) return;

    const wrapper = document.createElement("div");
    wrapper.className = "site-language-switcher";
    const label = document.createElement("label");
    label.textContent = `${labels[language.code].language}:`;
    const select = document.createElement("select");
    select.setAttribute("aria-label", labels[language.code].language);

    languages.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.segment ? `${item.segment}/${pagePath}` : pagePath;
      option.textContent = item.native;
      option.selected = item.code === language.code;
      select.appendChild(option);
    });

    select.addEventListener("change", () => {
      window.location.href = resolvedPath(`${base}${select.value}`);
    });
    label.appendChild(select);
    wrapper.appendChild(label);
    header.appendChild(wrapper);
  }

  function rewriteTopNavigation() {
    const tabs = document.querySelectorAll(".md-tabs a");
    if (tabs.length < 5) return;

    const paths = [
      "",
      "guides/1-data-preparation/",
      "glossary/",
      "contributing-translations/",
      "licence/"
    ];

    Array.from(tabs).slice(0, 5).forEach((tab, index) => {
      tab.textContent = topNavigation[language.code][index];
      tab.href = resolvedPath(`${base}${language.segment ? `${language.segment}/` : ""}${paths[index]}`);
    });
  }

  function rewriteTableOfContents() {
    document.querySelectorAll(".md-sidebar--secondary .md-nav__title").forEach((title) => {
      const icon = title.querySelector(".md-nav__icon");
      title.textContent = labels[language.code].toc;
      if (icon) title.appendChild(icon);
    });
  }

  function rewritePrimaryNavigation() {
    const list = document.querySelector(".md-nav--primary > .md-nav__list");
    if (!list) return;

    const primaryNav = list.parentElement;
    primaryNav.classList.add("site-language-nav");

    const languagePrefix = language.segment ? `${language.segment}/` : "";
    const utilityPaths = [
      "glossary/",
      "contributing-translations/",
      "licence/"
    ];
    const items = [];

    function linkItem(label, path, active) {
      const item = document.createElement("li");
      item.className = "md-nav__item";
      const link = document.createElement("a");
      link.className = `md-nav__link${active ? " md-nav__link--active" : ""}`;
      link.href = resolvedPath(`${base}${languagePrefix}${path}`);
      link.textContent = label;
      item.appendChild(link);
      return item;
    }

    items.push(linkItem(topNavigation[language.code][0], "", !pagePath));

    const guideItem = document.createElement("li");
    guideItem.className = "md-nav__item md-nav__item--nested";
    const toggleId = `site-language-guides-${language.code}`;
    const toggle = document.createElement("input");
    toggle.className = "md-nav__toggle md-toggle";
    toggle.type = "checkbox";
    toggle.id = toggleId;
    toggle.checked = true;
    const guideLabel = document.createElement("label");
    guideLabel.className = "md-nav__link";
    guideLabel.htmlFor = toggleId;
    guideLabel.id = `${toggleId}-label`;
    guideLabel.textContent = topNavigation[language.code][1];
    const guideIcon = document.createElement("span");
    guideIcon.className = "md-nav__icon md-icon";
    guideLabel.appendChild(guideIcon);
    guideItem.appendChild(toggle);
    guideItem.appendChild(guideLabel);
    const guideNav = document.createElement("nav");
    guideNav.className = "md-nav site-language-guide-nav";
    guideNav.dataset.mdLevel = "1";
    guideNav.setAttribute("aria-labelledby", `${toggleId}-label`);
    guideNav.setAttribute("aria-expanded", "true");
    const guideList = document.createElement("ul");
    guideList.className = "md-nav__list";
    guides.forEach((guide, index) => {
      const item = linkItem(guideTitles[language.code][index], guide, pagePath === guide);
      guideList.appendChild(item);
    });
    guideNav.appendChild(guideList);
    guideItem.appendChild(guideNav);
    items.push(guideItem);

    utilityPaths.forEach((path, index) => {
      items.push(linkItem(topNavigation[language.code][index + 2], path, false));
    });

    list.replaceChildren(...items);
  }

  function addSectionNavigation() {
    const content = document.querySelector(".md-content__inner");
    if (!content || !guides.includes(pagePath) || document.querySelector(".site-section-nav")) return;

    const index = guides.indexOf(pagePath);
    const nav = document.createElement("nav");
    nav.className = "site-section-nav";
    nav.setAttribute("aria-label", "Guide navigation");

    if (index > 0) {
      const previous = document.createElement("a");
      previous.href = resolvedPath(localizedPath(guides[index - 1]));
      previous.innerHTML = `<small>← ${labels[language.code].previous}</small><span>${guideTitles[language.code][index - 1]}</span>`;
      nav.appendChild(previous);
    } else {
      nav.appendChild(document.createElement("span"));
    }

    if (index < guides.length - 1) {
      const next = document.createElement("a");
      next.href = resolvedPath(localizedPath(guides[index + 1]));
      next.innerHTML = `<small>${labels[language.code].next} →</small><span>${guideTitles[language.code][index + 1]}</span>`;
      nav.appendChild(next);
    }

    content.appendChild(nav);
  }

  function init() {
    addLanguageSelector();
    rewriteTopNavigation();
    rewriteTableOfContents();
    rewritePrimaryNavigation();
    addSectionNavigation();
    document.body.classList.add("site-navigation-ready");
    window.setTimeout(rewriteTopNavigation, 0);
    window.setTimeout(rewriteTopNavigation, 250);
    window.setTimeout(rewritePrimaryNavigation, 0);
    window.setTimeout(rewritePrimaryNavigation, 250);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
