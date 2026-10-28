# Run from coconut-main/ in the coconut conda env (WSL).
# The script's defaults changed after this run. These extra flags restore the settings it
# used (the lens-word cleanup added later still changes the teacher texts slightly):
#   --nla-lens plain --nla-lens-tokens 10 --nla-explanation-tokens 48
#   --nla-repeat-window 0 --no-nla-distinct-lens
python -u nla.py \
  --nla-train-examples 800 --nla-val-examples 32 --nla-test-examples 500 \
  --nla-warmup-steps 300 --nla-steps 150 --nla-seed 0 \
  --nla-lens plain --nla-lens-tokens 10 --nla-explanation-tokens 48 \
  --nla-repeat-window 0 --no-nla-distinct-lens
