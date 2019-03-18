# Generate Thai Song Lyrics

We use a multi-layer RNN (LSTM) for a lyrics generation task.
See Jupyter notebook for data scraping, training, and some visualization.


## Dataset and model

We use the lyrics from [siamzone](https://www.siamzone.com) website and our training set. 
The training preparation is similar to this [blog post](https://brangerbriz.com/blog/using-machine-learning-to-create-new-melodies). 
Basically, we use previous context words to predict the next word and then update the weight of our LSTM model.


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
