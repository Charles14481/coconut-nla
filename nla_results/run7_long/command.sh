# Run from coconut-main/ in the coconut conda env (WSL). Same settings as run4_long, plus
# the added metrics (explanation_metrics at each validation pass and at test, splice_metrics at
# test) and the question-only AR baseline, trained for as many steps as the NLA's warm start.
# Everything else is nla.py's defaults (full teacher with plain + z-scored lens, 15 words
# each, 64-token explanations, repeat window 3, distinct lens, adapter, pooled AR readout,
# lr 1e-4 / RL 1e-6 / RL AR 1e-5).
# Don't set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: under WSL it crashed two attempts
# of run4_long with "CUDA driver error: device not ready".
python -u nla.py \
  --nla-train-examples 10000 --nla-val-examples 64 --nla-test-examples 500 \
  --nla-warmup-steps 9000 --nla-steps 2000 --nla-evals-per-phase 10 \
  --nla-question-steps 9000 --nla-seed 0
