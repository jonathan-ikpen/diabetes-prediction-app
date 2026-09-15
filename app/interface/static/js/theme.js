/**
 * Theme toggle and mobile navigation.
 *
 * Progressive enhancement only: every page is fully usable with JavaScript
 * disabled. This file adds the light/dark switch and the small-screen menu.
 */
(function () {
  'use strict';

  var STORAGE_KEY = 'diabetes-app-theme';
  var root = document.documentElement;

  /* ---------------------------------------------------------------- theme */

  function currentTheme() {
    return root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  function syncToggleIcon(theme) {
    document.querySelectorAll('[data-theme-icon]').forEach(function (element) {
      element.hidden = element.getAttribute('data-theme-icon') !== theme;
    });
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    syncToggleIcon(theme);
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch (error) {
      // Persisting is a convenience; the toggle still works without it.
    }
  }

  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    syncToggleIcon(currentTheme());
    toggle.addEventListener('click', function () {
      applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
    });
  }

  // Follow the operating system if the visitor has never chosen explicitly.
  var mediaQuery = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
  if (mediaQuery && typeof mediaQuery.addEventListener === 'function') {
    mediaQuery.addEventListener('change', function (event) {
      var hasChoice = false;
      try {
        hasChoice = Boolean(window.localStorage.getItem(STORAGE_KEY));
      } catch (error) {
        hasChoice = false;
      }
      if (!hasChoice) {
        var theme = event.matches ? 'dark' : 'light';
        root.setAttribute('data-theme', theme);
        syncToggleIcon(theme);
      }
    });
  }

  /* ----------------------------------------------------------- navigation */

  var navToggle = document.getElementById('nav-toggle');
  var nav = document.getElementById('site-nav');

  if (navToggle && nav) {
    navToggle.addEventListener('click', function () {
      var isOpen = nav.getAttribute('data-open') === 'true';
      var nextState = !isOpen;

      nav.setAttribute('data-open', String(nextState));
      navToggle.setAttribute('aria-expanded', String(nextState));

      document.querySelectorAll('[data-nav-icon]').forEach(function (element) {
        var wants = nextState ? 'open' : 'closed';
        element.hidden = element.getAttribute('data-nav-icon') !== wants;
      });
    });

    // Close the menu when a link is followed on a small screen.
    nav.addEventListener('click', function (event) {
      if (event.target.closest('a') && nav.getAttribute('data-open') === 'true') {
        navToggle.click();
      }
    });
  }
})();
