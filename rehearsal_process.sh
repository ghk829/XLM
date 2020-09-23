#!/bin/bash

FASTBPE=tools/fastBPE/fast
PAIR=de-en
CON_DOMAINS="subtitles-koran"
SRC_DOMAINS="wmt"

BASE_FILE="tok"

for SRC_DOMAIN in $(echo $SRC_DOMAINS | sed -e 's/\,/ /g'); do

  OUTPATH=data/processed/$PAIR/CON-$SRC_DOMAIN-$CON_DOMAINS
  mkdir -p $OUTPATH

  cp data/processed/$PAIR/$SRC_DOMAIN/vocab $OUTPATH/vocab
  cp data/processed/$PAIR/$SRC_DOMAIN/codes $OUTPATH/codes
  cp data/processed/$PAIR/$SRC_DOMAIN/en.vocab $OUTPATH/en.vocab
  cp data/processed/$PAIR/$SRC_DOMAIN/de.vocab $OUTPATH/de.vocab

  for SPLIT in "train"; do
    for LG in "en" "de"; do
      cp data/processed/$PAIR/$SRC_DOMAIN/$SPLIT.$PAIR.$LG $OUTPATH/$SPLIT.$PAIR.$LG
    done
  done

  for DOMAIN in $(echo $CON_DOMAINS | sed -e 's/\-/ /g'); do

    echo $BASE_FILE

    for SPLIT in "train"; do

      for LG in "en" "de"; do
        $FASTBPE applybpe $OUTPATH/$SPLIT.$PAIR.$LG.tmp dataset/$DOMAIN-$SPLIT.$BASE_FILE.$LG $OUTPATH/codes $OUTPATH/$LG.vocab
        echo "VOCAB IS USED $LG $SPLIT"
        cat $OUTPATH/$SPLIT.$PAIR.$LG.tmp >>$OUTPATH/$SPLIT.$PAIR.$LG
      done
    done
  done

  for SPLIT in "train"; do
    for LG in "en" "de"; do
      python preprocess.py $OUTPATH/vocab $OUTPATH/$SPLIT.$PAIR.$LG
    done
  done

done
