/**
 * Result page enhancement.
 *
 * The server already rendered a complete, deterministic explanation composed
 * from the WHO/IDF knowledge base, so this page is finished before any script
 * runs. This file asks /api/narrative for the richer language-model version and
 * swaps it in if one comes back.
 *
 * If the request fails, times out, or the provider is unconfigured, the
 * server-rendered text simply stays. The visitor is never left without an
 * explanation.
 */
(function () {
  'use strict';

  var panel = document.getElementById('narrative-panel');

  /* ------------------------------------------------------------- printing */

  var printButton = document.getElementById('print-result');
  if (printButton) {
    printButton.addEventListener('click', function () {
      window.print();
    });
  }

  if (!panel) {
    return;
  }

  var body = document.getElementById('narrative-body');
  var status = document.getElementById('narrative-status');
  var sourceLine = document.getElementById('narrative-source');
  var endpoint = panel.getAttribute('data-endpoint');

  var payload;
  try {
    payload = JSON.parse(panel.getAttribute('data-payload'));
  } catch (error) {
    return;
  }

  if (!payload || !endpoint || typeof window.fetch !== 'function') {
    return;
  }

  function setStatus(text, visible) {
    if (!status) {
      return;
    }
    status.hidden = !visible;
    var label = status.querySelector('[data-status-text]');
    if (label) {
      label.textContent = text;
    }
  }

  function render(narrative) {
    if (!body) {
      return;
    }

    var paragraphs = narrative.paragraphs && narrative.paragraphs.length
      ? narrative.paragraphs
      : [narrative.text];

    body.textContent = '';
    paragraphs.forEach(function (text) {
      var p = document.createElement('p');
      p.textContent = text;   // textContent, never innerHTML: model output is untrusted
      body.appendChild(p);
    });

    if (sourceLine) {
      sourceLine.textContent = narrative.is_ai_generated
        ? 'Written by ' + narrative.model + ' via ' + narrative.source
          + ', grounded in this tool\'s WHO/IDF knowledge base. Model-generated text can '
          + 'contain errors — confirm anything clinical with a healthcare professional.'
        : (narrative.fallback_reason || 'Composed from this tool\'s WHO/IDF knowledge base.');
    }

    setStatus(narrative.is_ai_generated ? 'AI written' : 'Knowledge base', true);
  }

  /* --------------------------------------------------------------- fetch */

  // Bound the wait so a slow provider cannot leave the panel spinning.
  var controller = typeof AbortController === 'function' ? new AbortController() : null;
  var timer = window.setTimeout(function () {
    if (controller) {
      controller.abort();
    }
  }, 45000);

  // The server-rendered knowledge-base explanation stays on screen while the
  // request is in flight. Hiding good content behind a skeleton to wait on an
  // optional upgrade would be a downgrade.
  setStatus('Writing a fuller explanation', true);

  window.fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: controller ? controller.signal : undefined
  })
    .then(function (response) {
      if (!response.ok) {
        throw new Error('Narrative request failed with status ' + response.status);
      }
      return response.json();
    })
    .then(function (data) {
      if (data && data.narrative) {
        render(data.narrative);
      }
    })
    .catch(function () {
      // Leave the server-rendered knowledge-base narrative exactly as it is.
      setStatus('Knowledge base', true);
    })
    .then(function () {
      window.clearTimeout(timer);
    });
})();
