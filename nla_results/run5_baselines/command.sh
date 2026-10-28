# Run from coconut-main/ in the coconut conda env (WSL). Small check of the added metrics:
# FVE vs the per-thought-index mean, raw-space FVE, and the question-only AR baseline
# (same AR architecture, start weights, lr and batch size as the NLA's, trained for as many
# steps as the NLA's warm start). Everything else is nla.py's defaults.
python -u nla.py \
  --nla-train-examples 300 --nla-val-examples 16 --nla-test-examples 100 \
  --nla-warmup-steps 300 --nla-steps 30 --nla-evals-per-phase 3 \
  --nla-question-steps 300 --nla-seed 0
