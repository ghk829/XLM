#!/bin/bash

FASTBPE=tools/fastBPE/fast
PAIR=de-en
SRC_DOMAIN="subtitles"
TGT_DOMAIN="koran"
OUTPATH=data/processed/$PAIR/$SRC_DOMAIN-$TGT_DOMAIN

mkdir -p $OUTPATH

cp data/processed/$PAIR/$TGT_DOMAIN/vocab $OUTPATH/vocab
cp data/processed/$PAIR/$TGT_DOMAIN/codes $OUTPATH/codes


BASE_FILE="tok"
NCODES=32000
for DOMAIN in $SRC_DOMAIN; do

  echo $BASE_FILE

  for SPLIT in "dev" "test"; do

    for LG in "en" "de"; do
      $FASTBPE applybpe $OUTPATH/$SPLIT.$PAIR.$LG dataset/$DOMAIN-$SPLIT.$BASE_FILE.$LG $OUTPATH/codes
    done


    for LG in "en" "de"; do
      $FASTBPE applybpe $OUTPATH/$SPLIT.$PAIR.$LG dataset/$DOMAIN-$SPLIT.$BASE_FILE.$LG $OUTPATH/codes $OUTPATH/$LG.vocab
      echo "VOCAB IS USED $LG $SPLIT"
    done

    for LG in "en" "de"; do
      python preprocess.py $OUTPATH/vocab $OUTPATH/$SPLIT.$PAIR.$LG
    done

  done
  # dev -> valid
  for LG in "en" "de"; do
    mv $OUTPATH/dev.$PAIR.$LG.pth $OUTPATH/valid.$PAIR.$LG.pth
  done

  # Monolingual
  for SPLIT in "valid" "test"; do
    cp $OUTPATH/$SPLIT.$PAIR.en.pth $OUTPATH/$SPLIT.en.pth
    cp $OUTPATH/$SPLIT.$PAIR.de.pth $OUTPATH/$SPLIT.de.pth
  done

done
