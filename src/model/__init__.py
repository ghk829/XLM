# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch

from .pretrain import load_embeddings
from .transformer import DECODER_ONLY_PARAMS, TransformerModel  # , TRANSFORMER_LAYER_PARAMS
from .memory import HashingMemory

logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.sample_alpha < 1
    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    # input sentence noise for DAE
    if len(params.ae_steps) == 0:
        assert params.word_shuffle == 0
        assert params.word_dropout == 0
        assert params.word_blank == 0
    else:
        assert params.word_shuffle == 0 or params.word_shuffle > 1
        assert 0 <= params.word_dropout < 1
        assert 0 <= params.word_blank < 1

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # memory
    if params.use_memory:
        HashingMemory.check_params(params)
        s_enc = [x for x in params.mem_enc_positions.split(',') if x != '']
        s_dec = [x for x in params.mem_dec_positions.split(',') if x != '']
        assert len(s_enc) == len(set(s_enc))
        assert len(s_dec) == len(set(s_dec))
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_enc)
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_dec)
        params.mem_enc_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_enc]
        params.mem_dec_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_dec]
        assert len(params.mem_enc_positions) + len(params.mem_dec_positions) > 0
        assert len(params.mem_enc_positions) == 0 or 0 <= min([x[0] for x in params.mem_enc_positions]) <= max([x[0] for x in params.mem_enc_positions]) <= params.n_layers - 1
        assert len(params.mem_dec_positions) == 0 or 0 <= min([x[0] for x in params.mem_dec_positions]) <= max([x[0] for x in params.mem_dec_positions]) <= params.n_layers - 1

    # reload pretrained word embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    # reload a pretrained model
    if params.reload_model != '':
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            s = params.reload_model.split(',')
            assert len(s) == 2
            assert all([x == '' or os.path.isfile(x) for x in s])


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


def build_model(params, dico):
    """
    Build model.
    """
    is_adapt = params.is_adapt
    if params.encoder_only:

        if params.lstm:
            from src.model.lstm import LSTMModel
            model = LSTMModel(len(dico.counts), params.emb_dim, len(dico.counts), dropout_p=params.dropout, n_layers=params.n_layers, max_length=512)
        else:

            # build
            model = TransformerModel(params, dico, is_encoder=True, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(model, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(params.local_rank))['model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            # # HACK to reload models with less layers
            # for i in range(12, 24):
            #     for k in TRANSFORMER_LAYER_PARAMS:
            #         k = k % i
            #         if k in model.state_dict() and k not in reloaded:
            #             logger.warning("Parameter %s not found. Ignoring ..." % k)
            #             reloaded[k] = model.state_dict()[k]

            model.load_state_dict(reloaded)

        logger.info("Model: {}".format(model))
        logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))

        return model.cuda()

    elif params.dual_encoder:
        encoder1 = TransformerModel(params, dico, is_encoder=True, with_output=True)
        encoder2 = TransformerModel(params, dico, is_encoder=True, with_output=True)

        # reload pretrained word embeddings : skip

        # reload a pretrained model
        if params.reload_model != '':
            enc_path1, enc_path2 = params.reload_model.split(',')
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(enc_path1, map_location=lambda storage, loc: storage.cuda(params.local_rank))['model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            encoder1.load_state_dict(reloaded)

            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(enc_path2, map_location=lambda storage, loc: storage.cuda(params.local_rank))['model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            encoder2.load_state_dict(reloaded)

        encoder2 = encoder1

        return encoder1.cuda(), encoder2.cuda()
    else:

        if params.local_adapt:
            from .transformer import MetaTransformerModel
            # build
            encoder = MetaTransformerModel(params, dico, is_encoder=True,
                                       with_output=True)
            decoder = MetaTransformerModel(params, dico, is_encoder=False, with_output=True)
        else:
            # build
            encoder = TransformerModel(params, dico, is_encoder=True, with_output=True)  # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
            decoder = TransformerModel(params, dico, is_encoder=False, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(encoder, dico, word2id, embeddings)
            set_pretrain_emb(decoder, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            enc_path, dec_path = params.reload_model.split(',')
            assert not (enc_path == '' and dec_path == '')

            # reload encoder
            if enc_path != '':
                logger.info("Reloading encoder from %s ..." % enc_path)
                enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
                if all([k.startswith('module.') for k in enc_reload.keys()]):
                    enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}

                if not is_adapt:
                    encoder.load_state_dict(enc_reload,strict=False)
                else:
                    enc_reload = {k: v for k, v in enc_reload.items() if
                                  k not in ['embeddings.weight', 'pred_layer.proj.weight', 'pred_layer.proj.bias']}
                    encoder.load_state_dict(enc_reload, strict=False)

                if params.freeze_heads != '':
                    for p in encoder.parameters():
                        p.requires_grad = False

                    freeze_heads = list(map(int, params.freeze_heads.split(',')))
                    unfreeze_heads = set(range(params.n_heads)) - set(freeze_heads)
                    unfreeze_heads = list(unfreeze_heads)
                    print('unfreeze_heads')
                    print(unfreeze_heads)
                    for unfreeze_head in unfreeze_heads:
                        encoder.attentions[len(encoder.attentions)-1].q_lin[unfreeze_head].weight.requires_grad = True
                        encoder.attentions[len(encoder.attentions)-1].q_lin[unfreeze_head].bias.requires_grad = True
                        encoder.attentions[len(encoder.attentions)-1].k_lin[unfreeze_head].weight.requires_grad = True
                        encoder.attentions[len(encoder.attentions)-1].k_lin[unfreeze_head].bias.requires_grad = True
                        encoder.attentions[len(encoder.attentions)-1].v_lin[unfreeze_head].weight.requires_grad = True
                        encoder.attentions[len(encoder.attentions)-1].v_lin[unfreeze_head].bias.requires_grad = True


            # reload decoder
            if dec_path != '':
                logger.info("Reloading decoder from %s ..." % dec_path)
                dec_reload = torch.load(dec_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                dec_reload = dec_reload['model' if 'model' in dec_reload else 'decoder']
                if all([k.startswith('module.') for k in dec_reload.keys()]):
                    dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}
                for i in range(params.n_layers):
                    for name in DECODER_ONLY_PARAMS:
                        if name % i not in dec_reload:
                            logger.warning("Parameter %s not found." % (name % i))
                            # for multisegmenthead
                            #dec_reload[name % i] = decoder.state_dict()[name % i]

                if not is_adapt:
                    decoder.load_state_dict(dec_reload, strict=False)
                else:
                    dec_reload = {k: v for k, v in dec_reload.items() if
                                  k not in ['embeddings.weight', 'pred_layer.proj.weight', 'pred_layer.proj.bias']}
                    decoder.load_state_dict(dec_reload, strict=False)

                if params.freeze_heads != '':
                    for p in decoder.parameters():
                        p.requires_grad = False

                    freeze_heads = list(map(int, params.freeze_heads.split(',')))
                    unfreeze_heads = set(range(params.n_heads)) - set(freeze_heads)
                    unfreeze_heads = list(unfreeze_heads)
                    logger.info('unfreeze_heads')
                    logger.info(unfreeze_heads)
                    for unfreeze_head in unfreeze_heads:
                        decoder.attentions[len(decoder.attentions)-1].q_lin[unfreeze_head].weight.requires_grad = True
                        decoder.attentions[len(decoder.attentions)-1].q_lin[unfreeze_head].bias.requires_grad = True
                        decoder.attentions[len(decoder.attentions)-1].k_lin[unfreeze_head].weight.requires_grad = True
                        decoder.attentions[len(decoder.attentions)-1].k_lin[unfreeze_head].bias.requires_grad = True
                        decoder.attentions[len(decoder.attentions)-1].v_lin[unfreeze_head].weight.requires_grad = True
                        decoder.attentions[len(decoder.attentions)-1].v_lin[unfreeze_head].bias.requires_grad = True

                    for unfreeze_head in unfreeze_heads:
                        decoder.encoder_attn[len(decoder.encoder_attn)-1].q_lin[unfreeze_head].weight.requires_grad = True
                        decoder.encoder_attn[len(decoder.encoder_attn)-1].q_lin[unfreeze_head].bias.requires_grad = True
                        decoder.encoder_attn[len(decoder.encoder_attn)-1].k_lin[unfreeze_head].weight.requires_grad = True
                        decoder.encoder_attn[len(decoder.encoder_attn)-1].k_lin[unfreeze_head].bias.requires_grad = True
                        decoder.encoder_attn[len(decoder.encoder_attn)-1].v_lin[unfreeze_head].weight.requires_grad = True
                        decoder.encoder_attn[len(decoder.encoder_attn)-1].v_lin[unfreeze_head].bias.requires_grad = True

        logger.debug("Encoder: {}".format(encoder))
        logger.debug("Decoder: {}".format(decoder))
        logger.info("Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        logger.info("Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))

        return encoder.cuda(), decoder.cuda()
