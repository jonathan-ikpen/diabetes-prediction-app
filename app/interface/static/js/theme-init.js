/**
 * Theme bootstrap.
 *
 * Runs synchronously in <head>, before the first paint, so a returning visitor
 * on the dark theme never sees a white flash. Kept in its own file rather than
 * an inline <script> so the page satisfies `script-src 'self'`.
 */
(function () {
  'use strict';

  var STORAGE_KEY = 'diabetes-app-theme';

  function resolveTheme() {
    try {
      var saved = window.localStorage.getItem(STORAGE_KEY);
      if (saved === 'light' || saved === 'dark') {
        return saved;
      }
    } catch (error) {
      // localStorage is unavailable in private mode on some browsers.
      // Fall through to the operating system preference.
    }

    var query = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
    return query && query.matches ? 'dark' : 'light';
  }

  document.documentElement.setAttribute('data-theme', resolveTheme());
})();
