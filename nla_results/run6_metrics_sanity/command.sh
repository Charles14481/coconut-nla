# Run from coconut-main/ in the coconut conda env (WSL). Small sanity check of the added
# metrics (explanation_metrics at every validation pass and at test, splice_metrics at test,
# question-only AR). Everything else is nla.py's defaults.
python -u nla.py \
  --nla-train-examples 300 --nla-val-examples 16 --nla-test-examples 100 \
  --nla-warmup-steps 300 --nla-steps 30 --nla-evals-per-phase 3 \
  --nla-question-steps 100 --nla-seed 0
