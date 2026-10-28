# Run from coconut-main/ in the coconut conda env (WSL).
# Note: this ran before the z-lens list started skipping words already in the plain list,
# so today's code builds slightly different teacher texts from the same command.
python -u nla.py \
  --nla-train-examples 2000 --nla-val-examples 32 --nla-test-examples 500 \
  --nla-warmup-steps 1000 --nla-steps 300 \
  --nla-lens both --nla-lens-tokens 15 --nla-explanation-tokens 64 --nla-repeat-window 3
