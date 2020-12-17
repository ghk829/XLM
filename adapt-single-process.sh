#!/bin/bash

FASTBPE=tools/fastBPE/fast
PAIR=en
EVAL_DOMAINS="subtitles koran it emea acquis"
SRC_DOMAINS="wmt"

BASE_FILE="tok"

for SRC_DOMAIN in $(echo $SRC_DOMAINS | sed -e 's/\,/ /g'); do

  for EVAL_DOMAIN in $(echo $EVAL_DOMAINS | sed -e 's/\,/ /g'); do

    OUTPATH=data/processed/$PAIR/ADAPT/$EVAL_DOMAIN
    mkdir -p $OUTPATH
    cp data/processed/$PAIR/$SRC_DOMAIN/vocab $OUTPATH/vocab
    cp data/processed/$PAIR/$SRC_DOMAIN/codes $OUTPATH/codes
    cp data/processed/$PAIR/$SRC_DOMAIN/en.vocab $OUTPATH/en.vocab

    echo $BASE_FILE

    for SPLIT in "train" "dev" "test"; do

      for LG in "en"; do
        $FASTBPE applybpe $OUTPATH/$SPLIT.$PAIR.$LG dataset/$EVAL_DOMAIN-$SPLIT.$BASE_FILE.$LG $OUTPATH/codes $OUTPATH/$LG.vocab
        echo "VOCAB IS USED $LG $SPLIT"
      done

      for LG in "en"; do
        python preprocess.py $OUTPATH/vocab $OUTPATH/$SPLIT.$PAIR.$LG
      done

    done
    # dev -> valid
    for LG in "en"; do
      mv $OUTPATH/dev.$PAIR.$LG.pth $OUTPATH/valid.$PAIR.$LG.pth
    done

    # Monolingual
    for SPLIT in "train" "valid" "test"; do
      cp $OUTPATH/$SPLIT.$PAIR.en.pth $OUTPATH/$SPLIT.en.pth
    done

  done
done
