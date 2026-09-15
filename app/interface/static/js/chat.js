/**
 * The floating assistant.
 *
 * The server keeps nothing: the whole history is posted with each turn and
 * then forgotten, which is the same promise the rest of the app makes about
 * health data.
 *
 * The conversation is held in `sessionStorage` so it survives navigating
 * between pages and reloading, but dies the moment the tab closes.
 * `localStorage` would have been the easier choice and the wrong one - it
 * would leave someone's health details sitting on a shared or public computer
 * indefinitely.
 *
 * Assistant text is written with textContent, never innerHTML: it is model
 * output, and model output is untrusted.
 */
(function () {
  'use strict';

  var root = document.getElementById('chatbot');
  if (!root) {
    return;               // no chat-capable provider configured
  }

  var STORE_KEY = 'diabetes-app-chat';

  /** Never replay more than this from storage, however long the tab lived. */
  var STORE_MAX_TURNS = 24;

  var launcher = document.getElementById('chatbot-launcher');
  var panel = document.getElementById('chatbot-panel');
  var closeBtn = document.getElementById('chatbot-close');
  var log = document.getElementById('chatbot-log');
  var form = document.getElementById('chatbot-form');
  var input = document.getElementById('chatbot-input');
  var send = document.getElementById('chatbot-send');
  var clearBtn = document.getElementById('chatbot-clear');

  var history = [];
  var busy = false;

  /* ------------------------------------------------------------- storage */

  /**
   * Every access is wrapped: sessionStorage throws outright in some private
   * browsing modes, and a chat widget must not take the page down with it.
   */
  function loadHistory() {
    try {
      var raw = window.sessionStorage.getItem(STORE_KEY);
      if (!raw) {
        return [];
      }
      var parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        return [];
      }
      return parsed.filter(function (turn) {
        return turn && (turn.role === 'user' || turn.role === 'assistant') &&
          typeof turn.content === 'string';
      }).slice(-STORE_MAX_TURNS);
    } catch (error) {
      return [];
    }
  }

  function saveHistory() {
    try {
      window.sessionStorage.setItem(
        STORE_KEY, JSON.stringify(history.slice(-STORE_MAX_TURNS)));
    } catch (error) {
      // Storage full or unavailable. The conversation still works in memory.
    }
  }

  function clearHistory() {
    history = [];
    try {
      window.sessionStorage.removeItem(STORE_KEY);
    } catch (error) {
      // Nothing to do; the in-memory history is already gone.
    }
  }

  /* ------------------------------------------------------------ open/close */

  function setOpen(open) {
    panel.hidden = !open;
    launcher.setAttribute('aria-expanded', String(open));
    var wants = open ? 'open' : 'closed';
    launcher.querySelectorAll('[data-launcher]').forEach(function (icon) {
      icon.hidden = icon.getAttribute('data-launcher') !== wants;
    });
    if (open) {
      input.focus();
    } else {
      launcher.focus();
    }
  }

  launcher.addEventListener('click', function () {
    setOpen(panel.hidden);
  });
  closeBtn.addEventListener('click', function () {
    setOpen(false);
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && !panel.hidden) {
      setOpen(false);
    }
  });

  /* -------------------------------------------------------------- messages */

  /**
   * Render one bubble.
   *
   * The text is split into paragraphs here rather than relying on `pre-wrap`.
   * Preserving whitespace meant a message's own source indentation showed up
   * on screen, and it made stray newlines from the model render as ragged
   * gaps. Single newlines inside a paragraph collapse to spaces, blank lines
   * start a new paragraph.
   */
  function addMessage(role, text) {
    var el = document.createElement('div');
    el.className = 'chatbot__msg chatbot__msg--' + (role === 'user' ? 'user' : 'bot');

    String(text).trim().split(/\n\s*\n/).forEach(function (block) {
      var line = block.replace(/\s+/g, ' ').trim();
      if (!line) {
        return;
      }
      var p = document.createElement('p');
      p.textContent = line;      // never innerHTML: this is model output
      el.appendChild(p);
    });

    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  }

  /**
   * The score, shown as its own element rather than left inside the reply.
   * The number comes from the model's JSON, so what is displayed is what the
   * Random Forest returned - not the assistant's retelling of it.
   */
  function addPrediction(prediction) {
    var band = (prediction.risk_band && prediction.risk_band.id) || 'moderate';
    var card = document.createElement('div');
    card.className = 'chatbot__result chatbot__result--' + band;

    var value = document.createElement('span');
    value.className = 'chatbot__result-value numeric';
    value.textContent = prediction.percentage + '%';

    var label = document.createElement('span');
    label.className = 'chatbot__result-band';
    label.textContent = (prediction.risk_band && prediction.risk_band.label) || '';

    card.appendChild(value);
    card.appendChild(label);
    log.appendChild(card);
    log.scrollTop = log.scrollHeight;
  }

  /**
   * Draw the log from scratch: the greeting, then whatever the tab already
   * said. Called on load and again after the conversation is cleared.
   *
   * Only the words are replayed. A previous score is not redrawn, because
   * that figure came from the model at the time and re-rendering it here
   * would be the interface asserting a result rather than reporting one.
   */
  function render() {
    log.textContent = '';

    var greeting = log.getAttribute('data-greeting');
    if (greeting) {
      addMessage('bot', greeting);
    }
    history.forEach(function (turn) {
      addMessage(turn.role, turn.content);
    });
    clearBtn.hidden = history.length === 0;
  }

  clearBtn.addEventListener('click', function () {
    clearHistory();
    render();
    input.focus();
  });

  history = loadHistory();
  render();

  function setBusy(state) {
    busy = state;
    send.disabled = state;
    input.disabled = state;
    send.textContent = state ? 'Thinking' : 'Send';
  }

  /* ---------------------------------------------------------------- submit */

  form.addEventListener('submit', function (event) {
    event.preventDefault();
    if (busy) {
      return;
    }

    var text = input.value.trim();
    if (!text) {
      return;
    }

    addMessage('user', text);
    history.push({ role: 'user', content: text });
    saveHistory();
    clearBtn.hidden = false;
    input.value = '';
    setBusy(true);

    var thinking = addMessage('bot', '…');
    thinking.classList.add('chatbot__msg--pending');

    window.fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history })
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error('chat failed');
        }
        return response.json();
      })
      .then(function (data) {
        thinking.remove();
        var reply = (data.reply || '').trim() ||
          'Sorry, I did not catch that. Could you say it another way?';
        addMessage('bot', reply);

        // Only a real reply becomes part of the history; a degraded notice is
        // an interface message, not something the model said.
        if (!data.degraded) {
          history.push({ role: 'assistant', content: reply });
          saveHistory();
        }
        if (data.prediction) {
          addPrediction(data.prediction);
        }
      })
      .catch(function () {
        thinking.remove();
        addMessage('bot',
          'I could not reach the assistant. The step-by-step check still works and does ' +
          'not need it.');
      })
      .then(function () {
        setBusy(false);
        input.focus();
      });
  });
})();
