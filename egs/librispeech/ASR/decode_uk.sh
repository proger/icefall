export PYTHONPATH=$HOME/k2/k2/python
export CUDA_VISIBLE_DEVICES=

./pruned_transducer_stateless5/decode.py \
  --epoch 1 \
  --avg 1 \
  --use-averaged-model False \
  --exp-dir pruned_transducer_stateless5/exp-uk \
  --decoding-method fast_beam_search \
  --num-encoder-layers 18 \
  --dim-feedforward 1024 \
  --nhead 4 \
  --encoder-dim 256 \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --bpe-model uk/data/lang_bpe_256/bpe.model \
  --lang-dir uk/data/lang_bpe_256

#--decoding-method greedy_search \