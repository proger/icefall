from pathlib import Path

import k2
import lhotse
from lhotse.dataset import K2SpeechRecognitionDataset, OnTheFlyFeatures, SingleCutSampler
from lhotse.features import Fbank, FbankConfig
from torch.utils.data import DataLoader
import torch
import math
import sentencepiece as spm

from icefall.checkpoint import load_checkpoint
from train import get_params
from train import get_transducer_model
from beam_search import Nbest, get_texts


def make_cut(cut_id, text):
    s,d,w,u,*_ = cut_id.split('-')
    utt_id = '-'.join([s,d,w,u])
    speaker = '-'.join([s,d,w])
    path = f'/tank/uk/{s}/{d}/{w}/{u}.wav'
    recording = lhotse.Recording.from_file(path, recording_id=utt_id)
    return (cut_id, lhotse.MonoCut(id=cut_id, start=0, duration=recording.duration, channel=0, supervisions=[
        lhotse.SupervisionSegment(id=cut_id, recording_id=utt_id, start=0, duration=recording.duration, channel=0,
                                  text=text, speaker=speaker)
    ], recording=recording))


def make_loader(cutset, max_duration=200, num_workers=0):
    dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(
            Fbank(FbankConfig(num_mel_bins=80))
        ),
        return_cuts=True,
    )
    sampler = SingleCutSampler(cutset, max_duration=max_duration, shuffle=False)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
    )
    return loader


def get_model():
    params = get_params()
    params.update({
        'feature_dim': 80, 'subsampling_factor': 4,
        'model_warm_step': 3000, 'start_epoch': 1,
        'context_size': 2, 'prune_range': 5,
        'lm_scale': 0.25, 'am_scale': 0.0,
        'simple_loss_scale': 0.5, 'seed': 42,
        'use_fp16': True, 'num_encoder_layers': 18,
        'dim_feedforward': 1024, 'nhead': 4,
        'encoder_dim': 256, 'decoder_dim': 512,
        'joiner_dim': 512,
        'dynamic_chunk_training': False,
        'causal_convolution': False,
        'simulate_streaming': False,
        'short_chunk_size': 25,
        'num_left_chunks': 4,
        'full_libri': True,
        'max_duration': 500,
        'bucketing_sampler': True, 'num_buckets': 30,
        'concatenate_cuts': False,
        'duration_factor': 1.0, 'gap': 1.0,
        'on_the_fly_feats': False, 'shuffle': True,
        'drop_last': True, 'return_cuts': True,
        'enable_spec_aug': True, 'spec_aug_time_warp_factor': 80,
        'enable_musan': True,
        'input_strategy': 'PrecomputedFeatures',
        'blank_id': 0, 'vocab_size': 250,
        'num_workers': 0,
        'left_context': 64})
    params.bpe_model = '../uk/data/lang_bpe_250/bpe.model'
    params.exp_dir = Path('exp-uk-shuf')
    params.manifest_dir = Path('../uk/data/fbank')

    params.beam = 20
    params.max_contexts = 8
    params.max_states = 64
    params.max_sym_per_frame = 1
    params.num_paths = 100
    params.nbest_scale = 0.5

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    model = get_transducer_model(params)
    load_checkpoint('exp-uk-shuf/pretrained.pt', model)
    model.eval()
    return model, sp, params

def encode(model, batch, params, device='cpu'):
    LOG_EPS = math.log(1e-10)
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    feature = torch.nn.functional.pad(
        feature,
        pad=(0, 0, 0, params.left_context),
        value=LOG_EPS,
    )
    feature_lens = feature_lens + params.left_context

    encoder_out_, encoder_out_lens = model.encoder(
        x=feature, x_lens=feature_lens
    )
    encoder_out = model.joiner.encoder_proj(encoder_out_)
    return encoder_out, encoder_out_lens

def make_trivial_graph(params):
    return k2.trivial_graph(
        params.vocab_size - 1, device='cpu'
    )

def fast_beam_search_lattice(model, encoder_out, encoder_out_lens, params, decoding_graph):
    context_size = model.decoder.context_size
    vocab_size = model.decoder.vocab_size

    temperature = 1

    B, T, C = encoder_out.shape

    config = k2.RnntDecodingConfig(
        vocab_size=vocab_size,
        decoder_history_len=context_size,
        beam=params.beam,
        max_contexts=params.max_contexts,
        max_states=params.max_states,
    )
    individual_streams = []
    for i in range(B):
        individual_streams.append(k2.RnntDecodingStream(decoding_graph))
    decoding_streams = k2.RnntDecodingStreams(individual_streams, config)

    for t in range(T):
        # shape is a RaggedShape of shape (B, context)
        # contexts is a Tensor of shape (shape.NumElements(), context_size)
        shape, contexts = decoding_streams.get_contexts()
        # `nn.Embedding()` in torch below v1.7.1 supports only torch.int64
        contexts = contexts.to(torch.int64)
        # decoder_out is of shape (shape.NumElements(), 1, decoder_out_dim)
        decoder_out = model.decoder(contexts, need_pad=False)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # current_encoder_out is of shape
        # (shape.NumElements(), 1, joiner_dim)
        # fmt: off
        current_encoder_out = torch.index_select(
            encoder_out[:, t:t + 1, :], 0, shape.row_ids(1).to(torch.int64)
        )
        # fmt: on
        logits = model.joiner(
            current_encoder_out.unsqueeze(2),
            decoder_out.unsqueeze(1),
            project_input=False,
        )
        logits = logits.squeeze(1).squeeze(1)
        log_probs = (logits / temperature).log_softmax(dim=-1)
        decoding_streams.advance(log_probs)

    decoding_streams.terminate_and_flush_to_streams()
    lattice = decoding_streams.format_output(encoder_out_lens.tolist())
    return lattice


def lattice_nbest_list(lattice, sp, params, *, n=-1):
    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=params.num_paths,
        use_double_scores=True,
        nbest_scale=params.nbest_scale,
    )

    # The following code is modified from nbest.intersect()
    word_fsa = k2.invert(nbest.fsa)
    if hasattr(lattice, "aux_labels"):
        # delete token IDs as it is not needed
        del word_fsa.aux_labels
    word_fsa.scores.zero_()
    word_fsa_with_epsilon_loops = k2.linear_fsa_with_self_loops(word_fsa)
    path_to_utt_map = nbest.shape.row_ids(1)

    if hasattr(lattice, "aux_labels"):
        # lattice has token IDs as labels and word IDs as aux_labels.
        # inv_lattice has word IDs as labels and token IDs as aux_labels
        inv_lattice = k2.invert(lattice)
        inv_lattice = k2.arc_sort(inv_lattice)
    else:
        inv_lattice = k2.arc_sort(lattice)

    if inv_lattice.shape[0] == 1:
        path_lattice = k2.intersect_device(
            inv_lattice,
            word_fsa_with_epsilon_loops,
            b_to_a_map=torch.zeros_like(path_to_utt_map),
            sorted_match_a=True,
        )
    else:
        path_lattice = k2.intersect_device(
            inv_lattice,
            word_fsa_with_epsilon_loops,
            b_to_a_map=path_to_utt_map,
            sorted_match_a=True,
        )

    # path_lattice has word IDs as labels and token IDs as aux_labels
    path_lattice = k2.top_sort(k2.connect(path_lattice))
    tot_scores = path_lattice.get_tot_scores(
        use_double_scores=True,
        log_semiring=True,  # Note: we always use True
    )
    # See https://github.com/k2-fsa/icefall/pull/420 for why
    # we always use log_semiring=True

    ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)

    # Now nbest.fsa.scores contains acoustic scores

    if n == 1:
        path = k2.index_fsa(nbest.fsa, ragged_tot_scores.argmax())
        hyp_tokens = get_texts(path)
        return [(-ragged_tot_scores.max().item(), sp.decode(hyp_tokens))]
    else:
        entries = []
        for i in range(ragged_tot_scores.tot_size(1)):
            path = k2.index_fsa(nbest.fsa, torch.tensor([i], dtype=torch.int32))
            hyp_tokens = get_texts(path)
            entries.append((-ragged_tot_scores[0][i].item(), sp.decode(hyp_tokens)))

        entries = sorted(entries, key=lambda x: x[0])
        if n > 1:
            entries = entries[:n]
        return entries