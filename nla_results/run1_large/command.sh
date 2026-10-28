# Run from coconut-main/ in the coconut conda env (WSL).
python -u nla.py \
  --nla-train-examples 5000 --nla-val-examples 100 --nla-test-examples 500 \
  --nla-warmup-steps 2000 --nla-steps 300 \
  --nla-batch-size 4 --nla-group-size 4 \
  --nla-av-lr 3e-5 --nla-ar-lr 3e-5 --nla-rl-lr 3e-6 \
  --nla-explanation-tokens 12 --nla-kl-coef 0.05 --nla-seed 0
