import argparse
import os
import torch


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--eval_domain", type=str, default="subtitles",
                        help="Experiment dump path")

    parser.add_argument("--pretrained_domain", type=str, default="koran",
                        help="Experiment dump path")

    return parser


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    eval_domain = os.path.join('data/processed/de-en', params.eval_domain)

    pretrained_domain_path = os.path.join('data/processed/de-en', params.pretrained_domain)
    src_pth = os.path.join(pretrained_domain_path, 'valid.de-en.de.pth')
    tgt_pth = os.path.join(pretrained_domain_path, 'valid.de-en.en.pth')

    enc_dico = torch.load(src_pth)['dico']
    dec_dico = torch.load(tgt_pth)['dico']

    # torch.save(enc_dico, os.path.join(params.eval_domain, 'enc_dico.pth'))
    # torch.save(dec_dico, os.path.join(params.eval_domain, 'dec_dico.pth'))

    pth_files = [f for f in os.listdir(eval_domain) if
                 f.endswith('pth') and f not in ['enc_dico.pth', 'dec_dico.pth','dico.pth']]

    for f in pth_files:

        state = torch.load(os.path.join(eval_domain,f))

        if f.endswith('de.pth'):
            state['dico'] = enc_dico
        else:
            state['dico'] = dec_dico

        torch.save(state, os.path.join(eval_domain, f))
