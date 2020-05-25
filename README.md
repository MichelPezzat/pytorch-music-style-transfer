# Many-to-Many Symbolic Multi-track Music Genre Transfer
This the pytorch implementation of StarGAN-based model to realize music style transfer between different musical domains with a single generator.

# Dependencies

* Python 3.6+
* pytorch 1.0
* librosa
* tensorboardX
* scikit-learn
* numpy 1.14.2
* pretty_midi 0.2.8
* pypianoroll 0.1.3

# Usage

## Dataset

Link for Desert Camel MIDI Dataset adapted for  the training and testing of the model: https://drive.google.com/open?id=1QZP1OCTZnAwasmsglbBxXpJs6C8kbT-A

### Train

```
python main.py
```



### Convert



```
python main.py --mode test --test_iters 200000 --src_style bossanova --trg_style "['rock','funk']"
```