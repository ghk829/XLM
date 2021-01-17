import torch
from .utils import to_cuda
from torch.nn import functional as F
from logging import getLogger

logger = getLogger()


def nograd(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


@nograd
def build_nmt_domain_feature(data, params, batches, dataset):
    from src.model import build_model
    import copy

    params = copy.copy(params)
    params.reload_model = params.build_nmt_domain_feature  # 'multi-domain.pth,multi-domain.pth'
    encoder, decoder = build_model(params, data['dico'])
    encoder.eval()
    decoder.eval()
    params.reload_model = params.build_nmt_base_feature  # 'best-test_de-en_mt_bleu.pth,best-test_de-en_mt_bleu.pth'
    base_encoder, base_decoder = build_model(params, data['dico'])
    base_encoder.eval()
    base_decoder.eval()
    qzs = torch.Tensor([])
    qzss = torch.Tensor([])
    for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        logger.info(len(batches))
        batch_length = int(len(batches) / 10)
        i = 0
        for sentence_ids in batches:
            i += 1
            logger.info(i)
            pos1 = dataset.pos1[sentence_ids]
            pos2 = dataset.pos2[sentence_ids]
            sent1 = dataset.batch_sentences([dataset.sent1[a:b] for a, b in pos1])
            sent2 = dataset.batch_sentences([dataset.sent2[a:b] for a, b in pos2])

            (x1, len1), (x2, len2) = sent1, sent2
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            enc1 = enc1.half() if params.fp16 else enc1
            dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

            word_scores, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)
            length_y = (len2 - 1).cpu().tolist()
            scores = torch.Tensor([torch.index_select(score, 0, ref) for score, ref in
                                   zip(F.log_softmax(word_scores, dim=-1), y)]).to(len2.device)
            domain_finetuned = torch.split(scores, length_y)

            enc1 = base_encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            enc1 = enc1.half() if params.fp16 else enc1
            dec2 = base_decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

            word_scores, loss = base_decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)
            length_y = (len2 - 1).cpu().tolist()
            scores = torch.Tensor(
                [torch.index_select(score, 0, ref) for score, ref in zip(F.log_softmax(word_scores, dim=-1), y)]).to(
                len2.device)
            domain_based = torch.split(scores, length_y)

            qz = torch.Tensor([((f - b).sum() / l).cpu() for f, b, l in zip(domain_finetuned, domain_based, length_y)])
            qzss = torch.cat((qzss, qz))
            if i % batch_length == 0:
                qzs = torch.cat((qzs, qzss))
                qzss = torch.Tensor([])
        if len(qzss) != 0:
            qzs = torch.cat((qzs, qzss))
    return qzs


@nograd
def build_nlm_domain_feature(data, params, batches, dataset):
    from src.model import build_model
    import copy
    import numpy as np

    params = copy.copy(params)
    params.reload_model = params.build_nlm_domain_feature  # 'multi-domain.pth,multi-domain.pth'
    encoder  = build_model(params, data['dico'])
    encoder.eval()
    params.reload_model = params.build_nlm_base_feature  # 'best-test_de-en_mt_bleu.pth,best-test_de-en_mt_bleu.pth'
    base_encoder  = build_model(params, data['dico'])
    base_encoder.eval()
    qzs = torch.Tensor([])
    qzss = torch.Tensor([])
    sents = []
    sent_last_index = 0
    for lang1, lang2 in set(params.mt_steps):

        # lang1_id = params.lang2id[lang1]
        # lang2_id = params.lang2id[lang2]
        logger.info(len(batches))
        batch_length = int(len(batches) / 10)
        i = 0
        for sentence_ids in batches:
            i += 1
            logger.info(i)

            # a = dataset.bptt * sentence_ids
            # b = dataset.bptt * (sentence_ids + 1)
            # x, lengths = torch.from_numpy(dataset.data[a:b].astype(np.int64)), dataset.lengths
            if lang1 == 'en':
                pos = dataset.pos1[sentence_ids]
                sent = dataset.batch_sentences([dataset.sent1[a:b] for a, b in pos])
            elif lang2 == 'en':
                pos = dataset.pos2[sentence_ids]
                sent = dataset.batch_sentences([dataset.sent2[a:b] for a, b in pos])
            x, lengths_list = sent
            positions = None
            langs = None

            alen = torch.arange(lengths_list.max(), dtype=torch.long, device=lengths_list.device)
            pred_mask = alen[:, None] < lengths_list[None] - 1
            if params.context_size > 0:  # do not predict without context
                pred_mask[:params.context_size] = 0
            y = x[1:].masked_select(pred_mask[:-1])
            assert pred_mask.sum().item() == y.size(0)

            x_input = x.permute(1,0).reshape(-1,1)
            x_input = x_input[x_input != data['dico'].pad_index].reshape(-1, 1)
            lengths = lengths_list.sum().reshape(1)
            lengths_list = lengths_list.tolist()

            x_input, x, lengths, langs, pred_mask, y = to_cuda(x_input, x, lengths, langs, pred_mask, y)

            qz1 = []
            qz2 = []
            tmp_tensor_list = []
            for x_batch in x_input.split(256):
                _lengths = torch.tensor(x_batch.shape[0], device=x_batch.device).reshape(-1)
                tmp_tensor = encoder('fwd', x=x_batch, lengths=_lengths, positions=positions, langs=langs, causal=True)
                tmp_tensor_list.append(tmp_tensor)
            tensor = torch.cat(tmp_tensor_list,dim=0)
            for xx, tt, mm in zip(x.permute(1, 0), tensor.split(lengths_list), pred_mask.permute(1, 0)):
                mm = mm.masked_select(xx != data['dico'].pad_index)
                xx = xx.masked_select(xx != data['dico'].pad_index)
                yy = xx[1:].masked_select(mm[:-1])
                word_scores, loss = encoder('predict', tensor=tt, pred_mask=mm.reshape(-1,1), y=yy, get_scores=True)
                xe_loss = loss.item() * len(yy)
                n_words = yy.size(0)
                domain_finetuned = np.exp(xe_loss / n_words)
                qz1.append(domain_finetuned)
                sents.append([data['dico'].id2word[word_id.item()] for word_id in yy])

            tmp_tensor_list = []
            for x_batch in x_input.split(256):
                _lengths = torch.tensor(x_batch.shape[0], device=x_batch.device).reshape(-1)
                tmp_tensor = base_encoder('fwd', x=x_batch, lengths=_lengths, positions=positions, langs=langs, causal=True)
                tmp_tensor_list.append(tmp_tensor)
            tensor = torch.cat(tmp_tensor_list,dim=0)
            for xx, tt, mm in zip(x.permute(1, 0), tensor.split(lengths_list), pred_mask.permute(1, 0)):
                mm = mm.masked_select(xx != data['dico'].pad_index)
                xx = xx.masked_select(xx != data['dico'].pad_index)
                yy = xx[1:].masked_select(mm[:-1])
                word_scores, loss = base_encoder('predict', tensor=tt, pred_mask=mm.reshape(-1,1), y=yy, get_scores=True)
                xe_loss = loss.item() * len(yy)
                n_words = yy.size(0)
                domain_based = np.exp(xe_loss / n_words)
                qz2.append(domain_based)

            qz1 = torch.Tensor(np.array(qz1))
            qz2 = torch.Tensor(np.array(qz2))
            qz = (qz1 - qz2)
            qzss = torch.cat((qzss, qz))
            if i % batch_length ==0:
                before_last_index = sent_last_index
                sent_last_index = len(sents)
                result = {'domain_feature': qzss, 'sents': sents[before_last_index:sent_last_index]}
                if params.build_output_path.endswith('pth'):
                    import os
                    build_output_path = os.dirname(params.build_output_path)
                else:
                    build_output_path = params.build_output_path
                print(f'{i}th saved')
                torch.save(result, f'{build_output_path}/{i}.pth')
                qzs = torch.cat((qzs,qzss))
                qzss = torch.Tensor([])

        if len(qzss) != 0:
            qzs = torch.cat((qzs, qzss))

    return qzs, sents