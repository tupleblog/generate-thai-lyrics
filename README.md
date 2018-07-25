# Generate Thai Song Lyrics

We use a multi-layer RNN (Elman, GRU, or LSTM) on a text generation task.
See more customization details and related publications 
on [`word_language_model`](https://github.com/pytorch/examples/tree/master/word_language_model). 


## Training from scratch

You can download lyrics CSV file and train model directly.

- Download Thai song' lyrics dataframe (we scraped lyrics from [siamzone.com](https://www.siamzone.com/music/lyric/))
- [train.py](https://github.com/tupleblog/generate-thai-lyrics/blob/master/train.py) will generate `corpus_lyrics.pkl` 
(takes time to tokenize depends on your machine speed)

```bash
wget https://s3-us-west-2.amazonaws.com/thai-corpus/lyric_dataframe.csv -O ./data/lyric_dataframe.csv # download scraped Thai songs' lyrics to data folder
python train.py --cuda --epochs 40 --tied --lr 0.02 # Train a tied LSTM on Thai lyrics with CUDA for 40 epochs, learning rate = 0.2
```

## Training from pre-computed corpus

Alternatively, you can download pre-computed corpus and train the model.

```bash
wget https://s3-us-west-2.amazonaws.com/thai-corpus/corpus_lyrics.pkl # corpus
python train.py --cuda --epochs 40 --tied --lr 0.02 # Train a tied LSTM on Thai lyrics with CUDA for 40 epochs, learning rate = 0.2
```

## Generate lyrics

To generate lyrics, run the following command.

```bash
python generate.py --temperature 0.8 --words 200 # Generate lyrics samples from the trained LSTM model.
```

## Pre-trained model

We already trained LSTM sequence prediction model on Thai song lyrics where you can download from Google drive.
Download model from [here](https://drive.google.com/file/d/1wTMCBB3Vrwstld-LBwYEF6nwFHyqLJT7/view?usp=sharing) and paste to `./thai-song-model.pt` 
and corpus from [here](https://s3-us-west-2.amazonaws.com/thai-corpus/corpus_lyrics.pkl) and paste to `./corpus_lyrics.pkl`.

Now, you can predict Thai lyrics using the following command

```bash
python generate.py --temperature 0.8 --checkpoint ./thai-song-model.pt --words 200
```

## Dependencies

- [torch](https://pytorch.org/)
- [deepcut](https://github.com/rkcosmos/deepcut)


## Members

- [titipata](https://github.com/titipata)
- [kittinan](https://github.com/kittinan)
