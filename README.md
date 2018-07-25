# Generate Thai Song Lyrics

We use a multi-layer RNN (Elman, GRU, or LSTM) on a text generation task.
See more customization details and related publications 
on [`word_language_model`](https://github.com/pytorch/examples/tree/master/word_language_model). 


## Training from scratch
- Download Thai song' lyrics dataframe
- [train.py](https://github.com/tupleblog/generate-thai-lyrics/blob/master/train.py) will generate corpus_lyrics.pkl (take time depends on your machine speed)
```bash
wget https://s3-us-west-2.amazonaws.com/thai-corpus/lyric_dataframe.csv -O ./data/lyric_dataframe.csv # download scraped Thai songs' lyrics to data folder
python train.py --cuda --epochs 40 --tied --lr 0.02 # Train a tied LSTM on Thai lyrics with CUDA for 40 epochs, learning rate = 0.2
```

## Training with pre-corpus

```bash
wget https://s3-us-west-2.amazonaws.com/thai-corpus/corpus_lyrics.pkl # corpus
python train.py --cuda --epochs 40 --tied --lr 0.02 # Train a tied LSTM on Thai lyrics with CUDA for 40 epochs, learning rate = 0.2
```

## Generate lyrics
```bash
python generate.py --temperature 0.8 --words 200 # Generate lyrics samples from the trained LSTM model.
```

## Pre-trained model
Download [here](https://drive.google.com/file/d/1wTMCBB3Vrwstld-LBwYEF6nwFHyqLJT7/view?usp=sharing) and paste to ./thai-song-model.pt

```bash
python generate.py --temperature 0.8 --checkpoint ./thai-song-model.pt --words 200
```

## Dependencies

- [torch](https://pytorch.org/)
- [deepcut](https://github.com/rkcosmos/deepcut)


## Members

- [titipata](https://github.com/titipata)
- [kittinan](https://github.com/kittinan)
