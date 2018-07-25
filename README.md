# Generate Thai Song Lyrics

We use a multi-layer RNN (Elman, GRU, or LSTM) on a text generation task.
See more customization details and related publications 
on [`word_language_model`](https://github.com/pytorch/examples/tree/master/word_language_model). 


## Training and generating lyrics

```bash
wget https://s3-us-west-2.amazonaws.com/thai-corpus/lyric_dataframe.csv # download scraped Thai songs' lyrics to data folder
mv lyric_dataframe.csv data/
wget https://s3-us-west-2.amazonaws.com/thai-corpus/corpus_lyrics.pkl # corpus
python train.py --cuda --epochs 40 --tied --lr 0.02 # Train a tied LSTM on Thai lyrics with CUDA for 40 epochs, learning rate = 0.2
python generate.py --temperature 0.8      # Generate lyrics samples from the trained LSTM model.
```

## Dependencies

- [torch](https://pytorch.org/)
- [deepcut](https://github.com/rkcosmos/deepcut)


## Members

- [titipata](https://github.com/titipata)
- [kittinan](https://github.com/kittinan)