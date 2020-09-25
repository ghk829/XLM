#!/bin/bash

FASTBPE=tools/fastBPE/fast
PAIR=de-en
EVAL_DOMAINS="subtitles koran it emea acquis"
SRC_DOMAINS="subtitles koran it emea acquis"

BASE_FILE="tok"
OUTPATH=data/coverage_test
mkdir -p $OUTPATH

for SRC_DOMAIN in $(echo $SRC_DOMAINS | sed -e 's/\,/ /g'); do

  for EVAL_DOMAIN in $(echo $EVAL_DOMAINS | sed -e 's/\,/ /g'); do

    SRC_CODES=data/processed/$PAIR/$SRC_DOMAIN/codes
    SRC_VOCAB_PATH=data/processed/$PAIR/$SRC_DOMAIN

    echo $BASE_FILE
    echo $SRC_DOMAIN-$EVAL_DOMAIN

    for SPLIT in "train"; do

      for LG in "en" "de"; do
        $FASTBPE applybpe $OUTPATH/$SRC_DOMAIN-$EVAL_DOMAIN.$SPLIT.$PAIR.$LG dataset/$EVAL_DOMAIN-$SPLIT.$BASE_FILE.$LG $SRC_CODES $SRC_VOCAB_PATH/$LG.vocab
        echo "VOCAB IS USED $LG $SPLIT"
      done

      for LG in "en" "de"; do
        python preprocess.py $SRC_VOCAB_PATH/vocab $OUTPATH/$SRC_DOMAIN-$EVAL_DOMAIN.$SPLIT.$PAIR.$LG
      done
    done
  done
done
