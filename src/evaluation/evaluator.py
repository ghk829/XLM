# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import subprocess
from collections import OrderedDict
import numpy as np
import torch

from ..utils import to_cuda, restore_segmentation, concat_batches
from ..model.memory import HashingMemory
from torch.autograd import grad
from ..model.data_actor import BaseActor

BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH)


logger = getLogger()


def kl_score(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    _x = x.copy()
    _x[x == 0] = 1
    return np.log(len(x)) + (x * np.log(_x)).sum()


def gini_score(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    B = np.cumsum(np.sort(x)).mean()
    return 1 - 2 * B


def tops(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    y = np.cumsum(np.sort(x))
    top50, top90, top99 = y.shape[0] - np.searchsorted(y, [0.5, 0.1, 0.01])
    return top50, top90, top99


def eval_memory_usage(scores, name, mem_att, mem_size):
    """
    Evaluate memory usage (HashingMemory / FFN).
    """
    # memory slot scores
    assert mem_size > 0
    mem_scores_w = np.zeros(mem_size, dtype=np.float32)  # weighted scores
    mem_scores_u = np.zeros(mem_size, dtype=np.float32)  # unweighted scores

    # sum each slot usage
    for indices, weights in mem_att:
        np.add.at(mem_scores_w, indices, weights)
        np.add.at(mem_scores_u, indices, 1)

    # compute the KL distance to the uniform distribution
    mem_scores_w = mem_scores_w / mem_scores_w.sum()
    mem_scores_u = mem_scores_u / mem_scores_u.sum()

    # store stats
    scores['%s_mem_used' % name] = float(100 * (mem_scores_w != 0).sum() / len(mem_scores_w))

    scores['%s_mem_kl_w' % name] = float(kl_score(mem_scores_w))
    scores['%s_mem_kl_u' % name] = float(kl_score(mem_scores_u))

    scores['%s_mem_gini_w' % name] = float(gini_score(mem_scores_w))
    scores['%s_mem_gini_u' % name] = float(gini_score(mem_scores_u))

    top50, top90, top99 = tops(mem_scores_w)
    scores['%s_mem_top50_w' % name] = float(top50)
    scores['%s_mem_top90_w' % name] = float(top90)
    scores['%s_mem_top99_w' % name] = float(top99)

    top50, top90, top99 = tops(mem_scores_u)
    scores['%s_mem_top50_u' % name] = float(top50)
    scores['%s_mem_top90_u' % name] = float(top90)
    scores['%s_mem_top99_u' % name] = float(top99)


class Evaluator(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.dico = data['dico']
        self.params = params
        self.memory_list = trainer.memory_list

        # create directory to store hypotheses, and reference files for BLEU evaluation
        if self.params.is_master:
            params.hyp_path = os.path.join(params.dump_path, 'hypotheses')
            subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()
            self.create_reference_files()

    def get_iterator(self, data_set, lang1, lang2=None, stream=False):
        """
        Create a new iterator for a dataset.
        """
        assert data_set in ['valid', 'test']
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None

        # hacks to reduce evaluation time when using many languages
        if len(self.params.langs) > 30:
            eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh", "ab", "ay", "bug", "ha", "ko", "ln", "min", "nds", "pap", "pt", "tg", "to", "udm", "uk", "zh_classical"])
            eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"])
            subsample = 10 if (data_set == 'test' or lang1 not in eval_lgs) else 5
            n_sentences = 600 if (data_set == 'test' or lang1 not in eval_lgs) else 1500
        elif len(self.params.langs) > 5:
            subsample = 10 if data_set == 'test' else 5
            n_sentences = 300 if data_set == 'test' else 1500
        else:
            # n_sentences = -1 if data_set == 'valid' else 100
            n_sentences = -1
            subsample = 1

        if lang2 is None:
            if stream:
                iterator = self.data['mono_stream'][lang1][data_set].get_iterator(shuffle=False, subsample=subsample)
            else:
                iterator = self.data['mono'][lang1][data_set].get_iterator(
                    shuffle=False,
                    group_by_size=True,
                    n_sentences=n_sentences,
                )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data['para'][(_lang1, _lang2)][data_set].get_iterator(
                shuffle=False,
                group_by_size=True,
                n_sentences=n_sentences
            )

        for batch in iterator:
            yield batch if lang2 is None or lang1 < lang2 else batch[::-1]

    def create_reference_files(self):
        """
        Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}

        for (lang1, lang2), v in self.data['para'].items():

            assert lang1 < lang2

            for data_set in ['valid', 'test']:

                # define data paths
                lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_set))
                lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set))

                # store data paths
                params.ref_paths[(lang2, lang1, data_set)] = lang1_path
                params.ref_paths[(lang1, lang2, data_set)] = lang2_path

                # text sentences
                lang1_txt = []
                lang2_txt = []

                # convert to text
                for (sent1, len1), (sent2, len2) in self.get_iterator(data_set, lang1, lang2):
                    lang1_txt.extend(convert_to_text(sent1, len1, self.dico, params))
                    lang2_txt.extend(convert_to_text(sent2, len2, self.dico, params))

                # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                # export hypothesis
                with open(lang1_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang1_txt) + '\n')
                with open(lang2_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang2_txt) + '\n')

                # restore original segmentation
                restore_segmentation(lang1_path)
                restore_segmentation(lang2_path)

    def mask_out(self, x, lengths, rng):
        """
        Decide of random words to mask out.
        We specify the random generator to ensure that the test is the same at each epoch.
        """
        params = self.params
        slen, bs = x.size()

        # words to predict - be sure there is at least one word per sentence
        to_predict = rng.rand(slen, bs) <= params.word_pred
        to_predict[0] = 0
        for i in range(bs):
            to_predict[lengths[i] - 1:, i] = 0
            if not np.any(to_predict[:lengths[i] - 1, i]):
                v = rng.randint(1, lengths[i] - 1)
                to_predict[v, i] = 1
        pred_mask = torch.from_numpy(to_predict.astype(np.uint8))

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_mask = _x_real.clone().fill_(params.mask_index)
        x = x.masked_scatter(pred_mask, _x_mask)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():

            for data_set in ['valid', 'test']:

                # causal prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.clm_steps:
                    self.evaluate_clm(scores, data_set, lang1, lang2)

                # prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.mlm_steps:
                    self.evaluate_mlm(scores, data_set, lang1, lang2)

                # machine translation task (evaluate perplexity and accuracy)
                for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):
                    eval_bleu = params.eval_bleu and params.is_master
                    self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu)

                # report average metrics per language
                _clm_mono = [l1 for (l1, l2) in params.clm_steps if l2 is None]
                if len(_clm_mono) > 0:
                    scores['%s_clm_ppl' % data_set] = np.mean([scores['%s_%s_clm_ppl' % (data_set, lang)] for lang in _clm_mono])
                    scores['%s_clm_acc' % data_set] = np.mean([scores['%s_%s_clm_acc' % (data_set, lang)] for lang in _clm_mono])
                _mlm_mono = [l1 for (l1, l2) in params.mlm_steps if l2 is None]
                if len(_mlm_mono) > 0:
                    scores['%s_mlm_ppl' % data_set] = np.mean([scores['%s_%s_mlm_ppl' % (data_set, lang)] for lang in _mlm_mono])
                    scores['%s_mlm_acc' % data_set] = np.mean([scores['%s_%s_mlm_acc' % (data_set, lang)] for lang in _mlm_mono])

        return scores

    def evaluate_clm(self, scores, data_set, lang1, lang2):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None
        l1l2 = lang1 if lang2 is None else f"{lang1}-{lang2}"

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        for batch in self.get_iterator(data_set, lang1, lang2, stream=(lang2 is None)):

            # batch
            if lang2 is None:
                x, lengths = batch
                positions = None
                langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
            else:
                (sent1, len1), (sent2, len2) = batch
                x, lengths, positions, langs = concat_batches(sent1, len1, lang1_id, sent2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=True)

            # words to predict
            alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
            pred_mask = alen[:, None] < lengths[None] - 1
            y = x[1:].masked_select(pred_mask[:-1])
            assert pred_mask.sum().item() == y.size(0)

            # cuda
            x, lengths, positions, langs, pred_mask, y = to_cuda(x, lengths, positions, langs, pred_mask, y)

            # forward / loss
            tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=True)
            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()
            if eval_memory:
                for k, v in self.memory_list:
                    all_mem_att[k].append((v.last_indices, v.last_scores))

        # log
        logger.info("Found %i words in %s. %i were predicted correctly." % (n_words, data_set, n_valid))

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_clm_ppl' % (data_set, l1l2)
        acc_name = '%s_%s_clm_acc' % (data_set, l1l2)
        scores[ppl_name] = np.exp(xe_loss / n_words)
        scores[acc_name] = 100. * n_valid / n_words

        # compute memory usage
        if eval_memory:
            for mem_name, mem_att in all_mem_att.items():
                eval_memory_usage(scores, '%s_%s_%s' % (data_set, l1l2, mem_name), mem_att, params.mem_size)

    def evaluate_mlm(self, scores, data_set, lang1, lang2):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.encoder
        model.eval()
        model = model.module if params.multi_gpu else model

        rng = np.random.RandomState(0)

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None
        l1l2 = lang1 if lang2 is None else f"{lang1}_{lang2}"

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        for batch in self.get_iterator(data_set, lang1, lang2, stream=(lang2 is None)):

            # batch
            if lang2 is None:
                x, lengths = batch
                positions = None
                langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
            else:
                (sent1, len1), (sent2, len2) = batch
                x, lengths, positions, langs = concat_batches(sent1, len1, lang1_id, sent2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=True)

            # words to predict
            x, y, pred_mask = self.mask_out(x, lengths, rng)

            # cuda
            x, y, pred_mask, lengths, positions, langs = to_cuda(x, y, pred_mask, lengths, positions, langs)

            # forward / loss
            tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += len(y)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()
            if eval_memory:
                for k, v in self.memory_list:
                    all_mem_att[k].append((v.last_indices, v.last_scores))

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_mlm_ppl' % (data_set, l1l2)
        acc_name = '%s_%s_mlm_acc' % (data_set, l1l2)
        scores[ppl_name] = np.exp(xe_loss / n_words) if n_words > 0 else 1e9
        scores[acc_name] = 100. * n_valid / n_words if n_words > 0 else 0.

        # compute memory usage
        if eval_memory:
            for mem_name, mem_att in all_mem_att.items():
                eval_memory_usage(scores, '%s_%s_%s' % (data_set, l1l2, mem_name), mem_att, params.mem_size)


class SingleEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build language model evaluator.
        """
        super().__init__(trainer, data, params)
        self.model = trainer.model


class EncDecEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder

    def evaluate_mt(self, scores, data_set, lang1, lang2, eval_bleu):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs

        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        for batch in self.get_iterator(data_set, lang1, lang2):

            # generate batch
            (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            # encode source sentence
            if params.l0_weight != 0:
                enc1, _reg_loss = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            else:
                enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            enc1 = enc1.half() if params.fp16 else enc1

            if params.l0_weight != 0 and params.dec_self:
                dec2, _reg_loss = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1,
                                               src_len=len1)
            else:
                dec2 = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

            # loss
            word_scores, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()
            if eval_memory:
                for k, v in self.memory_list:
                    all_mem_att[k].append((v.last_indices, v.last_scores))

            # generate translation - translate / convert to text
            if eval_bleu:
                max_len = int(1.5 * len1.max().item() + 10)
                if max_len > 512:
                    continue # maximum generation length
                if params.beam_size == 1:
                    generated, lengths = decoder.generate(enc1, len1, lang2_id, max_len=max_len)
                else:
                    generated, lengths = decoder.generate_beam(
                        enc1, len1, lang2_id, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping,
                        max_len=max_len
                    )
                hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mt_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_mt_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

        # compute memory usage
        if eval_memory:
            for mem_name, mem_att in all_mem_att.items():
                eval_memory_usage(scores, '%s_%s-%s_%s' % (data_set, lang1, lang2, mem_name), mem_att, params.mem_size)

        # compute BLEU
        if eval_bleu:

            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_set)]

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mt_bleu' % (data_set, lang1, lang2)] = bleu


def convert_to_text(batch, lengths, dico, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + '0')
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1


class MultiDomainEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.domains = params.domains
        self.p = params.prior_ratios
        self.params = params

        data_actor = BaseActor(len(params.domains))
        self.data_actor = data_actor.cuda()
        self.data_optimizer = torch.optim.Adam([p for p in self.data_actor.parameters() if p.requires_grad],
                                               lr=params.data_actor_lr)

    def create_reference_files(self):
        """
        Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}

        for (lang1, lang2, domain), v in self.data['para'].items():

            assert lang1 < lang2

            for data_set in ['valid', 'test']:

                # define data paths
                lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.{3}.txt'.format(lang2, lang1, data_set,domain))
                lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.{3}.txt'.format(lang1, lang2, data_set,domain))

                # store data paths
                params.ref_paths[(lang2, lang1, data_set,domain)] = lang1_path
                params.ref_paths[(lang1, lang2, data_set,domain)] = lang2_path

                # text sentences
                lang1_txt = []
                lang2_txt = []

                # convert to text
                for (sent1, len1), (sent2, len2) in self.get_iterator(data_set, lang1, lang2,domain):
                    lang1_txt.extend(convert_to_text(sent1, len1, self.dico, params))
                    lang2_txt.extend(convert_to_text(sent2, len2, self.dico, params))

                # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                # export hypothesis
                with open(lang1_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang1_txt) + '\n')
                with open(lang2_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang2_txt) + '\n')

                # restore original segmentation
                restore_segmentation(lang1_path)
                restore_segmentation(lang2_path)

    def get_iterator(self, data_set, lang1, lang2, domain):
        """
        Create a new iterator for a dataset.
        """
        assert data_set in ['valid', 'test']
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs

        n_sentences = -1
        subsample = 1

        _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
        iterator = self.data['para'][(_lang1, _lang2,domain)][data_set].get_iterator(
            shuffle=False,
            group_by_size=True,
            n_sentences=n_sentences,
            eval=True
        )

        for batch in iterator:
            yield batch if lang2 is None or lang1 < lang2 else batch[::-1]

    def evaluate_mt(self, scores, data_set, lang1, lang2, eval_bleu,domain):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs

        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        for batch in self.get_iterator(data_set, lang1, lang2,domain):

            # generate batch
            (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            # encode source sentence
            if params.l0_weight != 0:
                enc1, _reg_loss = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            else:
                enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            enc1 = enc1.half() if params.fp16 else enc1

            if params.l0_weight != 0 and params.dec_self:
                dec2, _reg_loss = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1,
                                               src_len=len1)
            else:
                dec2 = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

            # loss
            word_scores, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()
            if eval_memory:
                for k, v in self.memory_list:
                    all_mem_att[k].append((v.last_indices, v.last_scores))

            # generate translation - translate / convert to text
            if eval_bleu:
                max_len = int(1.5 * len1.max().item() + 10)
                if max_len > 512:
                    continue # maximum generation length
                if params.beam_size == 1:
                    generated, lengths = decoder.generate(enc1, len1, lang2_id, max_len=max_len)
                else:
                    generated, lengths = decoder.generate_beam(
                        enc1, len1, lang2_id, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping,
                        max_len=max_len
                    )
                hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mt_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_mt_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

        # compute memory usage
        if eval_memory:
            for mem_name, mem_att in all_mem_att.items():
                eval_memory_usage(scores, '%s_%s-%s_%s' % (data_set, lang1, lang2, mem_name), mem_att, params.mem_size)

        # compute BLEU
        if eval_bleu:

            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}-{2}.{3}.{4}.txt'.format(scores['epoch'], lang1, lang2, data_set,domain)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_set,domain)]

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s-%s_mt_bleu' % (data_set, lang1, lang2,domain)] = bleu

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():
            for domain in params.domains:
                for data_set in ['valid', 'test']:

                    # causal prediction task (evaluate perplexity and accuracy)
                    for lang1, lang2 in params.clm_steps:
                        self.evaluate_clm(scores, data_set, lang1, lang2)

                    # prediction task (evaluate perplexity and accuracy)
                    for lang1, lang2 in params.mlm_steps:
                        self.evaluate_mlm(scores, data_set, lang1, lang2)

                    # machine translation task (evaluate perplexity and accuracy)
                    for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):
                        eval_bleu = params.eval_bleu and params.is_master
                        self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu,domain)

                    # report average metrics per language
                    _clm_mono = [l1 for (l1, l2) in params.clm_steps if l2 is None]
                    if len(_clm_mono) > 0:
                        scores['%s_clm_ppl' % data_set + domain] = np.mean([scores['%s_%s_clm_ppl' % (data_set, lang)] for lang in _clm_mono])
                        scores['%s_clm_acc' % data_set + domain] = np.mean([scores['%s_%s_clm_acc' % (data_set, lang)] for lang in _clm_mono])
                    _mlm_mono = [l1 for (l1, l2) in params.mlm_steps if l2 is None]
                    if len(_mlm_mono) > 0:
                        scores['%s_mlm_ppl' % data_set + domain] = np.mean([scores['%s_%s_mlm_ppl' % (data_set, lang)] for lang in _mlm_mono])
                        scores['%s_mlm_acc' % data_set + domain] = np.mean([scores['%s_%s_mlm_acc' % (data_set, lang)] for lang in _mlm_mono])

        return scores

    def mt_step_by_domain(self, lang1, lang2, batch):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """

        params = self.params

        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        encoder.train()
        decoder.train()

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # generate batch
        (x1, len1), (x2, len2) = batch
        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        # cuda
        x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

        # encode source sentence
        enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
        enc1 = enc1.transpose(0, 1)

        # decode target sentence
        dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)


        # loss
        _, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        #self.stats[('AE-%s' % lang1) if lang1 == lang2 else ('MT-%s-%s' % (lang1, lang2))].append(loss.item())

        return loss


    def get_enc_by_domain(self, lang1, lang2, batch):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """

        params = self.params

        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        encoder.train()
        decoder.train()

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # generate batch
        (x1, len1), (x2, len2) = batch
        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        # cuda
        x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

        # encode source sentence
        enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
        enc1 = enc1.transpose(0, 1)

        return enc1


    def get_iterator_by_sample(self, data_set, lang1, lang2, domain, n_samples=-1):
        """
        Create a new iterator for a dataset.
        """

        _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
        iterator = self.data['para'][(_lang1, _lang2,domain)][data_set].get_iterator(
            shuffle=True,
            group_by_size=True,
            n_sentences=n_samples
        )

        for batch in iterator:
            yield batch if lang2 is None or lang1 < lang2 else batch[::-1]

    def get_grad_sim(self, grad1,grad2):

        cosine_prod, cosine_norm, dev_cosine_norm = 0, 0, 0

        assert len(grad1)==len(grad2)
        grad1 = ( g1 for g1 in grad1 if g1 is not None )
        grad2 = ( g2 for g2 in grad2 if g2 is not None )


        for g1, g2 in zip(grad1,grad2):
            cosine_prod += (g1 * g2).sum().item()
            cosine_norm += g1.norm(2) ** 2
            dev_cosine_norm += g2.norm(2) ** 2

        cosine_sim = cosine_prod / ((cosine_norm * dev_cosine_norm) ** 0.5 + 1e-10)
        return cosine_sim.item(), cosine_norm, dev_cosine_norm

    def update_sampling_distribution(self, logits):
        for i, l in enumerate(logits):
            if logits[i] < 0:
                logits[i] = 0
        if sum(logits) == 0:
            logits = [0.1 for _ in range(len(logits))]
        p = np.array(logits) / sum(logits)
        # self.alpha_p == 0 in the paper
        self.p = p
        logger.info("final probs")
        logger.info(self.p)


    def update_dataset_ratio(self,trainer):

        if type(trainer) == torch.nn.parallel.DistributedDataParallel:
            trainer.module.p = self.p
        else:
            trainer.p = self.p
        # mainly de-en testing.
        #lang1, lang2 = self.params.mt_steps[0]

        #for ratio, domain in zip(self.p,self.domains):
        #    trainer.data['para'][(lang1, lang2, domain)]['train'].ratio = ratio

    def reset_dataset_ratio(self,trainer):

        data_actor = BaseActor(len(self.params.domains))
        self.data_actor = data_actor.cuda()
        self.data_optimizer = torch.optim.Adam([p for p in self.data_actor.parameters() if p.requires_grad],
                                               lr=self.params.data_actor_lr)

        self.p = [1 / len(self.params.domains) for _ in self.params.domains]

        if type(trainer) == torch.nn.parallel.DistributedDataParallel:
            trainer.module.p = self.p
        else:
            trainer.p = self.p

    def update_language_sampler_multidomain(self):
        """Update the distribution to sample languages """
        # calculate gradient direction
        # calculate dev grad
        # Initialize dev data iterator
        from itertools import chain
        # #dev dataset x #train dataset
        all_sim_list = []

        # mainly de-en testing.
        lang1, lang2 = self.params.mt_steps[0]

        encoder = self.encoder.module if self.params.multi_gpu else self.encoder
        decoder = self.decoder.module if self.params.multi_gpu else self.decoder

        for domain in self.domains:

            num_of_sample = 8
            train_set = self.get_iterator_by_sample('train',lang1,lang2, domain,num_of_sample)
            train_batch = next(train_set)

            train_loss = self.mt_step_by_domain(lang1,lang2,train_batch)

            g_train = grad(train_loss, chain(encoder.parameters(),decoder.parameters()),allow_unused=True)
            g_train = [ g for g in g_train if g is not None ]
            g_dev = []
            sim_list = []
            for valid_domain in self.domains:

                valid_set = self.get_iterator_by_sample('valid',lang1,lang2, valid_domain,num_of_sample)
                valid_batch = next(valid_set)

                valid_loss = self.mt_step_by_domain(lang1, lang2, valid_batch)
                tmp_g_dev = grad(valid_loss, chain(encoder.parameters(),decoder.parameters()),allow_unused=True)
                tmp_g_dev = [g for g in tmp_g_dev if g is not None]
                if len(g_dev) > 0:
                    g_dev = [ g_1 + g_2 for g_1, g_2 in zip(g_dev, tmp_g_dev)]
                else:
                    g_dev = tmp_g_dev
                sim, *_ = self.get_grad_sim(g_dev,g_train)
                sim_list.append(sim)
            all_sim_list.append(sim_list)
            torch.cuda.empty_cache()
        # ave
        sim_list = np.mean(np.array(all_sim_list), axis=0).tolist()

        feature = torch.ones(1,len(self.domains))
        grad_scale = torch.FloatTensor(sim_list).view(1, -1)

        feature = feature.cuda()
        grad_scale = grad_scale.cuda()

        for _ in range(self.params.data_actor_optim_step):
            a_logits = self.data_actor.forward(feature)
            loss = -torch.nn.functional.log_softmax(a_logits, dim=-1)
            if self.params.scale_reward:
                loss = loss * torch.softmax(a_logits, dim=-1).data
            loss = (loss * grad_scale).sum()
            loss.backward()
            self.data_optimizer.step()
            self.data_optimizer.zero_grad()
        with torch.no_grad():
            a_logits = self.data_actor.forward(feature)
            prob = torch.nn.functional.softmax(a_logits, dim=-1)
            sim_list = [i for i in prob.data.view(-1).cpu().numpy()]

        self.update_sampling_distribution(sim_list)
        #self.update_dataset_ratio()


    def update_language_sampler_multidomain(self):
        """Update the distribution to sample languages """
        # calculate gradient direction
        # calculate dev grad
        # Initialize dev data iterator
        from itertools import chain
        # #dev dataset x #train dataset
        all_sim_list = []

        # mainly de-en testing.
        lang1, lang2 = self.params.mt_steps[0]

        encoder = self.encoder.module if self.params.multi_gpu else self.encoder
        decoder = self.decoder.module if self.params.multi_gpu else self.decoder

        for domain in self.domains:

            num_of_sample = 8
            train_set = self.get_iterator_by_sample('train',lang1,lang2, domain,num_of_sample)
            train_batch = next(train_set)

            train_enc = self.get_enc_by_domain(lang1,lang2,train_batch)
            g_dev = []
            sim_list = []
            for valid_domain in self.domains:

                valid_set = self.get_iterator_by_sample('valid',lang1,lang2, valid_domain,num_of_sample)
                valid_batch = next(valid_set)

                valid_enc = self.get_enc_by_domain(lang1, lang2, valid_batch)

                sim = torch.nn.functional.cosine_similarity(train_enc.mean(dim=1),valid_enc.mean(dim=1),dim=1)
                sim_list.append(sim.cpu().detach().numpy())
            all_sim_list.append(sim_list)
            torch.cuda.empty_cache()
        # ave
        sim_list = np.array(all_sim_list).mean(axis=-1).mean(axis=0).tolist()

        feature = torch.ones(1,len(self.domains))
        grad_scale = torch.FloatTensor(sim_list).view(1, -1)

        feature = feature.cuda()
        grad_scale = grad_scale.cuda()

        for _ in range(self.params.data_actor_optim_step):
            a_logits = self.data_actor.forward(feature)
            loss = -torch.nn.functional.log_softmax(a_logits, dim=-1)
            if self.params.scale_reward:
                loss = loss * torch.softmax(a_logits, dim=-1).data
            loss = (loss * grad_scale).sum()
            loss.backward()
            self.data_optimizer.step()
            self.data_optimizer.zero_grad()
        with torch.no_grad():
            a_logits = self.data_actor.forward(feature)
            prob = torch.nn.functional.softmax(a_logits, dim=-1)
            sim_list = [i for i in prob.data.view(-1).cpu().numpy()]

        self.update_sampling_distribution(sim_list)

from ..model.transformer import construct_fast_params, deconstruct_fast_params, new_fast_params

class MetaMultiDomainEvaluator(MultiDomainEvaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)

        self.update_rate = 0.001
        self.inner_loop = 2



    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})

        for domain in params.domains:

            for data_set in ['valid', 'test']:
                if data_set == 'valid':
                    # machine translation task (evaluate perplexity and accuracy)
                    for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):
                        eval_bleu = params.eval_bleu and params.is_master
                        self.meta_evaluate_mt(scores, data_set, lang1, lang2, eval_bleu, domain)

                with torch.no_grad():

                    # causal prediction task (evaluate perplexity and accuracy)
                    for lang1, lang2 in params.clm_steps:
                        self.evaluate_clm(scores, data_set, lang1, lang2)

                    # prediction task (evaluate perplexity and accuracy)
                    for lang1, lang2 in params.mlm_steps:
                        self.evaluate_mlm(scores, data_set, lang1, lang2)

                    # machine translation task (evaluate perplexity and accuracy)
                    for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):
                        eval_bleu = params.eval_bleu and params.is_master
                        self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu,domain)

                    # report average metrics per language
                    _clm_mono = [l1 for (l1, l2) in params.clm_steps if l2 is None]
                    if len(_clm_mono) > 0:
                        scores['%s_clm_ppl' % data_set + domain] = np.mean([scores['%s_%s_clm_ppl' % (data_set, lang)] for lang in _clm_mono])
                        scores['%s_clm_acc' % data_set + domain] = np.mean([scores['%s_%s_clm_acc' % (data_set, lang)] for lang in _clm_mono])
                    _mlm_mono = [l1 for (l1, l2) in params.mlm_steps if l2 is None]
                    if len(_mlm_mono) > 0:
                        scores['%s_mlm_ppl' % data_set + domain] = np.mean([scores['%s_%s_mlm_ppl' % (data_set, lang)] for lang in _mlm_mono])
                        scores['%s_mlm_acc' % data_set + domain] = np.mean([scores['%s_%s_mlm_acc' % (data_set, lang)] for lang in _mlm_mono])

        return scores

    def meta_evaluate_mt(self, scores, data_set, lang1, lang2, eval_bleu,domain):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs

        self.encoder.train()
        self.decoder.train()

        from torch import optim

        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        optimizer_parameters = [p for p in encoder.parameters() if p.requires_grad] + [p for p in
                                                                                            decoder.parameters() if
                                                                                            p.requires_grad]
        self.meta_optim = optim.Adam(optimizer_parameters, lr=0.0001)

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        losses = []

        for batch in self.get_iterator(data_set, lang1, lang2,domain):

            # generate batch
            (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            def get_grad(g):
                if g is None:
                    return 0
                return g

            encoder_named_parameters = [(n, p) for n, p in encoder.named_parameters()]
            decoder_named_parameters = [(n, p) for n, p in decoder.named_parameters()] + [
                ('pred_layer.proj.weight', decoder.pred_layer.proj.weight)]  # bug fix
            encoder_fast_params = construct_fast_params(encoder_named_parameters)
            decoder_fast_params = construct_fast_params(decoder_named_parameters)
            encoder_named_params = [n for n, p in encoder_named_parameters]
            decoder_named_params = [n for n, p in decoder_named_parameters]

            for i in range(self.inner_loop):

                enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False, params=encoder_fast_params)
                enc1 = enc1.transpose(0, 1)
                dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1,
                               params=decoder_fast_params)
                _, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False,
                                  params=decoder_fast_params['pred_layer']['proj'])

                encoder_parameters = deconstruct_fast_params(encoder_fast_params, encoder_named_params)
                decoder_parameters = deconstruct_fast_params(decoder_fast_params, decoder_named_params)

                encoder_grads = grad(loss, encoder_parameters, allow_unused=True, retain_graph=True)
                decoder_grads = grad(loss, decoder_parameters)

                encoder_named_parameters = [(n, p) for n, p in zip(encoder_named_params, encoder_parameters)]
                decoder_named_parameters = [(n, p) for n, p in zip(decoder_named_params, decoder_parameters)]

                #encoder_named_parameters = new_fast_params(encoder_named_parameters, encoder_grads, self.update_rate)
                encoder_named_parameters = [(n, p - self.update_rate * get_grad(g)) for (n, p), g in zip(encoder_named_parameters, encoder_grads)]
                encoder_fast_params = construct_fast_params(encoder_named_parameters)
                #decoder_named_parameters = new_fast_params(decoder_named_parameters, decoder_grads, self.update_rate)
                decoder_named_parameters = [(n, p - self.update_rate * get_grad(g)) for (n, p), g in zip(decoder_named_parameters, decoder_grads)]
                decoder_fast_params = construct_fast_params(decoder_named_parameters)

                if i == self.inner_loop - 1:
                    torch.cuda.empty_cache()

                    enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False, params=encoder_fast_params)
                    enc1 = enc1.transpose(0, 1)
                    dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1,
                                   params=decoder_fast_params)
                    _, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False,
                                      params=decoder_fast_params['pred_layer']['proj'])

                    # g_train = grad(loss, chain(encoder.parameters(),decoder.parameters()),allow_unused=True)
                    self.meta_optim.zero_grad()
                    loss.backward()
                    self.meta_optim.step()

                    del encoder_parameters
                    del decoder_parameters
                    del encoder_grads
                    del decoder_grads
                    del encoder_named_parameters
                    del decoder_named_parameters
                    del encoder_fast_params
                    del decoder_fast_params
                    del loss
                    torch.cuda.empty_cache()


class DualEncoderEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.encoder1 = trainer.encoder1
        self.encoder2 = trainer.encoder2

    def evaluate_mt(self, scores, data_set, lang1, lang2, eval_bleu):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        pass