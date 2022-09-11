export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONPATH=$HOME/k2/k2/python

./pruned_transducer_stateless5/train.py \
  --world-size 2 \
  --num-epochs 30 \
  --start-epoch 1 \
  --full-libri 1 \
  --exp-dir pruned_transducer_stateless5/exp-uk-shuf \
  --max-duration 500 \
  --use-fp16 1 \
  --num-encoder-layers 18 \
  --dim-feedforward 1024 \
  --nhead 4 \
  --encoder-dim 256 \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --bpe-model uk/data/lang_bpe_250/bpe.model

#  --start-batch 24000 \
