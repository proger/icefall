export PYTHONPATH=$HOME/k2/k2/python
export CUDA_VISIBLE_DEVICES=0,1


./pruned_transducer_stateless5/decode.py \
  --epoch 27 \
  --avg 15 \
  --use-averaged-model True \
  --exp-dir pruned_transducer_stateless5/exp-uk-shuf \
  --decoding-method fast_beam_search \
  --num-encoder-layers 18 \
  --dim-feedforward 1024 \
  --nhead 4 \
  --encoder-dim 256 \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --bpe-model uk/data/lang_bpe_250/bpe.model \
  --lang-dir uk/data/lang_bpe_250

