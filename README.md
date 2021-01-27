# Generate Thai Song Lyrics

We use a transformer model trained on Siamzone lyrics to generate
new lyrics. The demo can be found on [https://tupleblog.github.io/generate-thai-lyrics/](https://tupleblog.github.io/generate-thai-lyrics/)

## Dataset and model

We use the lyrics from [siamzone](https://www.siamzone.com) website as our training set.
The training preparation is similar to this [blog post](https://brangerbriz.com/blog/using-machine-learning-to-create-new-melodies).
Basically, our task is to predict the next word given all the previous words.
You can see more script in `train` folder to see how we train LSTM or Transformer model.

## Next steps

We plan to improve our model using larger model. Please stay tuned!

## Dependencies

- [torch](https://pytorch.org/)
- [pythainlp](https://github.com/PyThaiNLP/pythainlp)
- [scikit-learn](https://scikit-learn.org/stable/)

## Members

This work is done by `tupleteam`

- [titipata](https://github.com/titipata)
- [kittinan](https://github.com/kittinan)
- [bachkukkik](https://github.com/bachkukkik)
- [tulakann](https://github.com/bluenex)
