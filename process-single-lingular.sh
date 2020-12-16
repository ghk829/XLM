#!/bin/bash

FASTBPE=tools/fastBPE/fast
PAIR=en
BASE_FILE="tok"
SRC_DOMAINS="wmt subtitles koran it emea acquis"
NCODES=32000
for DOMAIN in $(echo $SRC_DOMAINS | sed -e 's/\,/ /g'); do

  echo $BASE_FILE
  OUTPATH=data/processed/$PAIR/$DOMAIN
  mkdir -p $OUTPATH

  for LG in "en"; do
    for SPLIT in "train" "dev" "test"; do
      cat dataset/$DOMAIN-$SPLIT.$BASE_FILE.$LG >>$OUTPATH/$DOMAIN-all.$LG
    done
  done

  $FASTBPE learnbpe $NCODES $OUTPATH/$DOMAIN-all.en $OUTPATH/$DOMAIN-all.de >$OUTPATH/codes

  for SPLIT in "train" "dev" "test"; do

    for LG in "en"; do

      if [ $SPLIT = "train" ]; then
        $FASTBPE applybpe $OUTPATH/$SPLIT.$PAIR.$LG dataset/$DOMAIN-$SPLIT.$BASE_FILE.$LG $OUTPATH/codes
        $FASTBPE getvocab $OUTPATH/$SPLIT.$PAIR.en > $OUTPATH/vocab
        $FASTBPE getvocab $OUTPATH/$SPLIT.$PAIR.en > $OUTPATH/en.vocab
        echo "VOCAB IS BUILT"
      else
        for LG in "en"; do
          $FASTBPE applybpe $OUTPATH/$SPLIT.$PAIR.$LG dataset/$DOMAIN-$SPLIT.$BASE_FILE.$LG $OUTPATH/codes $OUTPATH/$LG.vocab
          echo "VOCAB IS USED $LG $SPLIT"
        done
      fi
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
