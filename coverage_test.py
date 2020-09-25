import torch



if __name__ == '__main__':

    domains = ['subtitles','koran','it','emea','acquis']

    for src_domain in domains:
        for tat_domain in domains:
            for lg in ['de','en']:
                data = torch.load(f'{src_domain}-{tat_domain}.train.de-en.{lg}.pth')
                coverage = 100. * ( 1- sum(data['unk_words'].values()) / (len(data['sentences']) - len(data['positions'])))
                print("################### COVERAGE ###########################")
                print(f'{src_domain}-{tat_domain}-{lg}')
                print(coverage)
                print('\n')

