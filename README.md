# Kaggle Competition: [Quick, Draw! Doodle Recognition Challenge](https://www.kaggle.com/c/quickdraw-doodle-recognition)

## How accurately can you identify a doodle?

![altext](https://storage.googleapis.com/kaggle-media/competitions/quickdraw/what-does-a-bee-look-like-1.png)

"Quick, Draw!" was released as an experimental game to educate the public in a playful way about how AI works. The game prompts users to draw an image depicting a certain category, such as ”banana,” “table,” etc. The game generated more than 1B drawings, of which a subset was publicly released as the basis for this competition’s training set. That subset contains 50M drawings encompassing 340 label categories.

Sounds fun, right? Here's the challenge: since the training data comes from the game itself, drawings can be incomplete or may not match the label. You’ll need to build a recognizer that can effectively learn from this noisy data and perform well on a manually-labeled test set from a different distribution.

Your task is to build a better classifier for the existing Quick, Draw! dataset. By advancing models on this dataset, Kagglers can improve pattern recognition solutions more broadly. This will have an immediate impact on handwriting recognition and its robust applications in areas including OCR (Optical Character Recognition), ASR (Automatic Speech Recognition) & NLP (Natural Language Processing).

## Main architecture
Inception V3:
* Input shape: 128 x 128
* Batch Size: 680



## Dependencies
The code was developed with the following configuration:
* python 3.6.5
* opencv 3.4.0
* numpy 1.14.2
* keras 2.2.4
* pandas 0.23.4
* matplotlib 2.2.2
* tensorflow 1.7

Other configuration will reasonably work

## Authors
* **Hoang Tung Lam** - [lamhoangtung](https://lab.zinza.com.vn/lamht)

