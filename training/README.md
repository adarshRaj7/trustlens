# Splice-detector training pipeline

No public splice-detection dataset (e.g. CASIA v2) was freely available, so
labeled data is synthesized from your own royalty-free photos.

## Run it

1. Drop 4+ diverse JPEGs into `sources/` (not committed — bring your own; keep
   licensing clean since this repo is public).
2. `pip install -r ../backend/requirements.txt`
3. `python train_eval.py`

This prints two numbers — **use the leave-one-photo-out number, not the
random-split one** (see the script's docstring for why) — and overwrites
`../backend/weights/splice_classifier.pkl`.

More/more-diverse source photos is the single highest-leverage way to improve
the held-out AUC; see `../backend/weights/SPLICE_MODEL_CARD.md` for current
numbers and methodology.
