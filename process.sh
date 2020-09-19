#!/bin/bash

FASTBPE=tools/fastBPE/fast
PAIR=en-de
BASE_FILE="tok"
NCODES=60000

for DOMAIN in "it" "emea" "acquis" "koran" "subtitles"; do

  OUTPATH=data/processed/$PAIR/$DOMAIN
  mkdir -p $OUTPATH

  for LG in "en" "de"; do
    for SPLIT in "train" "dev" "test"; do
      cat dataset/$DOMAIN-$SPLIT.$BASE_FILE.$LG >>$OUTPATH/$DOMAIN-all.$LG
    done
  done

  $FASTBPE learnbpe $NCODES $OUTPATH/$DOMAIN-all.en $OUTPATH/$DOMAIN-all.de >$OUTPATH/codes

  for SPLIT in "train" "dev" "test"; do

    for LG in "en" "de"; do
      $FASTBPE applybpe $OUTPATH/$SPLIT.$PAIR.$LG dataset/$DOMAIN-$SPLIT.$BASE_FILE.$LG $OUTPATH/codes
    done

    if [ $SPLIT = "train" ]; then
      $FASTBPE getvocab $OUTPATH/$SPLIT.$PAIR.en $OUTPATH/$SPLIT.$PAIR.de >$OUTPATH/vocab
      echo "VOCAB IS BUILT"
    fi

    for LG in "en" "de"; do
      python preprocess.py $OUTPATH/vocab $OUTPATH/$SPLIT.$PAIR.$LG
    donec

  done
  # prefix : dev -> valid
  for LG in "en" "de"; do
    mv $OUTPATH/dev.$PAIR.$LG.pth $OUTPATH/valid.$PAIR.$LG.pth
  done

  # XLM bug
  for SPLIT in "train" "valid" "test"; do
    mv $OUTPATH/$SPLIT.$PAIR.en.pth $OUTPATH/$SPLIT.de-en.en.pth
    mv $OUTPATH/$SPLIT.$PAIR.de.pth $OUTPATH/$SPLIT.de-en.de.pth
  done

  # Monolingual Corpus
  for SPLIT in "train" "valid" "test"; do
    cp $OUTPATH/$SPLIT.de-en.en.pth $OUTPATH/$SPLIT.en.pth
    cp $OUTPATH/$SPLIT.de-en.de.pth $OUTPATH/$SPLIT.de.pth
  done

done
