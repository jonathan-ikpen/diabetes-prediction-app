"""
Application source.

Holds the three layers the running web app is built from:

    interface/      what the user sees: routes, API, templates, CSS, JS
    engine/         the thinking: validation, model scoring, explanation
    knowledgebase/  the medical facts: WHO / IDF thresholds

Each layer only depends on the one below it. The training pipeline (`ml/`)
lives outside this package because it never runs during a web request.
"""
