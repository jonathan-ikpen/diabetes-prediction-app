/**
 * Assessment form enhancement.
 *
 * Adds inline validation, sample-profile filling, and a submit busy state.
 * The form is a plain HTML POST underneath, so it still works with JavaScript
 * disabled - the server validates every field regardless of what happens here.
 */
(function () {
  'use strict';

  var form = document.getElementById('assessment-form');
  if (!form) {
    return;
  }

  var inputs = Array.prototype.slice.call(form.querySelectorAll('input[type="number"]'));

  /* ------------------------------------------------------------ validation */

  /**
   * Mirrors the server rules in engine/validation.py. Client-side checks are a
   * convenience for the visitor; the server is the authority.
   */
  function validationMessage(input) {
    var raw = input.value.trim();
    var label = input.getAttribute('data-label') || 'This measurement';

    if (raw === '') {
      return 'This measurement is required.';
    }

    var value = Number(raw.replace(',', '.'));
    if (!isFinite(value)) {
      return 'Enter a number, using digits only.';
    }

    if (input.getAttribute('data-integer') === 'true' && !Number.isInteger(value)) {
      return 'Enter a whole number.';
    }

    var min = Number(input.getAttribute('data-min'));
    var max = Number(input.getAttribute('data-max'));
    if (value < min || value > max) {
      return label + ' must be between ' + min + ' and ' + max + '.';
    }

    return '';
  }

  function showError(input, message) {
    var slot = document.getElementById(input.id + '-error');
    if (slot) {
      slot.textContent = message;
    }
    if (message) {
      input.setAttribute('aria-invalid', 'true');
    } else {
      input.removeAttribute('aria-invalid');
    }
  }

  inputs.forEach(function (input) {
    // Validate on blur, then live-correct once the field is known to be bad,
    // so the visitor is not scolded mid-keystroke on first entry.
    input.addEventListener('blur', function () {
      showError(input, validationMessage(input));
    });

    input.addEventListener('input', function () {
      if (input.getAttribute('aria-invalid') === 'true') {
        showError(input, validationMessage(input));
      }
    });
  });

  /* ---------------------------------------------------------- sample fills */

  document.querySelectorAll('[data-sample]').forEach(function (button) {
    button.addEventListener('click', function () {
      var values;
      try {
        values = JSON.parse(button.getAttribute('data-sample'));
      } catch (error) {
        return;
      }

      Object.keys(values).forEach(function (name) {
        var input = form.querySelector('[name="' + name + '"]');
        if (input) {
          input.value = values[name];
          showError(input, '');
        }
      });

      form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  /* --------------------------------------------------------------- submit */

  form.addEventListener('submit', function (event) {
    var firstInvalid = null;

    inputs.forEach(function (input) {
      var message = validationMessage(input);
      showError(input, message);
      if (message && !firstInvalid) {
        firstInvalid = input;
      }
    });

    if (firstInvalid) {
      event.preventDefault();
      firstInvalid.focus();
      firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }

    // Scoring plus a possible LLM call can take a moment; show that it started
    // and prevent a double submission.
    var button = document.getElementById('submit-button');
    if (button) {
      button.setAttribute('aria-disabled', 'true');
      document.querySelectorAll('[data-submit-label]').forEach(function (element) {
        element.hidden = element.getAttribute('data-submit-label') !== 'busy';
      });
    }
  });

  /* ---------------------------------------------------------------- reset */

  form.addEventListener('reset', function () {
    inputs.forEach(function (input) {
      showError(input, '');
    });
  });
})();
