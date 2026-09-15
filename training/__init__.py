"""
Training pipeline: how the model gets made.

Nothing here runs when someone visits the website. This package exists to turn
the raw dataset into `training/model/model.pkl`, which the app then loads.

Run in this order:

    python -m training.fetch_dataset   download the data and verify it
    python -m training.preprocess      clean it       -> training/data/processed/
    python -m training.train           train it       -> training/model/model.pkl
    python -m training.evaluate        score the saved model and print a report

Everything the pipeline needs and produces lives in this folder: the scripts,
the data (`data/`) and the finished model (`model/`).
"""
