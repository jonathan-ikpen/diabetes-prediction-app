/**
 * The step-by-step interview.
 *
 * Progressive enhancement: the page ships every question in one ordinary form,
 * which works with scripting disabled. This file turns it into one question at
 * a time with a progress bar, and skips questions whose precondition was not
 * met (there is no point asking how many pregnancies if the answer was "no").
 */
(function () {
  'use strict';

  var form = document.getElementById('interview');
  if (!form) {
    return;
  }

  var steps = Array.prototype.slice.call(form.querySelectorAll('[data-step]'));
  var bar = document.getElementById('interview-bar');
  var fill = document.getElementById('interview-fill');
  var count = document.getElementById('interview-count');
  var backBtn = document.getElementById('interview-back');
  var nextBtn = document.getElementById('interview-next');
  var submitBtn = document.getElementById('interview-submit');

  var index = 0;

  /* ------------------------------------------------------------ conditions */

  function answerOf(id) {
    var checked = form.querySelector('input[name="' + id + '"]:checked');
    if (checked) {
      return checked.value;
    }
    var field = form.querySelector('[name="' + id + '"]');
    return field ? field.value.trim() : '';
  }

  /** A step is live only when its precondition is currently satisfied. */
  function isLive(step) {
    var dependsOn = step.getAttribute('data-depends-on');
    if (!dependsOn) {
      return true;
    }
    return answerOf(dependsOn) === step.getAttribute('data-depends-value');
  }

  function liveSteps() {
    return steps.filter(isLive);
  }

  /* ------------------------------------------------------------ validation */

  function validate(step) {
    var slot = step.querySelector('[data-error]');
    var message = '';

    var radios = step.querySelectorAll('input[type="radio"]');
    if (radios.length) {
      if (!step.querySelector('input[type="radio"]:checked')) {
        message = 'Please choose one of the options.';
      }
    } else {
      var input = step.querySelector('input[type="number"]');
      if (input) {
        var raw = input.value.trim();
        var min = Number(input.getAttribute('data-min'));
        var max = Number(input.getAttribute('data-max'));
        if (raw === '') {
          message = 'Please enter a number to continue.';
        } else if (!isFinite(Number(raw))) {
          message = 'Please enter a number, using digits only.';
        } else if (Number(raw) < min || Number(raw) > max) {
          message = 'Please enter a value between ' + min + ' and ' + max + '.';
        }
      }
    }

    if (slot) {
      slot.textContent = message;
    }
    return message === '';
  }

  /* --------------------------------------------------------------- display */

  function render() {
    var live = liveSteps();

    // A skipped precondition can strand the pointer past the end.
    if (index > live.length - 1) {
      index = live.length - 1;
    }
    if (index < 0) {
      index = 0;
    }

    steps.forEach(function (step) {
      step.hidden = step !== live[index];
    });

    var isLast = index === live.length - 1;
    backBtn.hidden = index === 0;
    nextBtn.hidden = isLast;
    submitBtn.hidden = !isLast;

    var pct = Math.round((index / live.length) * 100);
    fill.style.width = pct + '%';
    count.textContent = 'Question ' + (index + 1) + ' of ' + live.length;

    var field = live[index].querySelector('input');
    if (field && field.type === 'number') {
      field.focus();
    }
  }

  /* --------------------------------------------------------------- wiring */

  nextBtn.addEventListener('click', function () {
    var live = liveSteps();
    if (!validate(live[index])) {
      return;
    }
    index += 1;
    render();
  });

  backBtn.addEventListener('click', function () {
    index -= 1;
    render();
  });

  form.addEventListener('submit', function (event) {
    var live = liveSteps();
    for (var i = 0; i < live.length; i += 1) {
      if (!validate(live[i])) {
        index = i;
        render();
        event.preventDefault();
        return;
      }
    }
    submitBtn.setAttribute('aria-disabled', 'true');
  });

  // Choosing an option advances automatically - it is the whole point of a
  // one-question-at-a-time interview that it should feel like a conversation.
  form.addEventListener('change', function (event) {
    if (event.target.type !== 'radio') {
      return;
    }
    var live = liveSteps();
    if (live[index] && live[index].contains(event.target) && index < live.length - 1) {
      window.setTimeout(function () {
        index += 1;
        render();
      }, 180);
    } else {
      render();
    }
  });

  // Enter should move on, not submit the whole form early.
  form.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' && event.target.type === 'number') {
      event.preventDefault();
      if (!nextBtn.hidden) {
        nextBtn.click();
      } else {
        submitBtn.click();
      }
    }
  });

  bar.hidden = false;
  render();
})();
