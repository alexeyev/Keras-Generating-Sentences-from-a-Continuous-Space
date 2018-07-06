# Generating Sentences from a Continuous Space

Keras implementation of LSTM variational autoencoder based on the code 
in [twairball's repo](https://github.com/twairball/keras_lstm_vae). Totally rewritten. Doesn't follow the paper exactly,
but the main ideas are implemented.

## Quick start

To run a toy example, download [this small dataset](http://www.manythings.org/anki/fra-eng.zip) and 
unpack `fra.txt` into the the folder `./data/`. Then run
```bash
    $ python3 train.py
```

## References
    - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
    - [Generating-Sentences-from-a-Continuous-Space paper](https://arxiv.org/abs/1511.06349)
    - Architecture fixed and inference implemented thanks to [this article on seq2seq in Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)
 
## License
MIT

## TODO
    - Dropout and other tricks from the paper
    - Initialization with word2vec/GloVE/whatever using the Embedding layer and its weights matrix
 
## Citation

Not neccessary, but is greatly appreciated, if you use this work.

```bibtex

@misc{Alekseev2018lstmvaekeras,
  author = {Alekseev~A.M.},
  title = {Generating Sentences from a Continuous Space, Keras implementation.},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/alexeyev/Keras-Generating-Sentences-from-a-Continuous-Space}},
  commit = {the latest commit of the codebase you have used}
}

```
## Examples

Travelling the space: 

```
    1000 samples, 40 epochs, toy example: train data
    ==  	 i 'm lucky . 	 	 ==
    1.00	 i 'm lucky 
    0.83	 i 'm lucky 
    0.67	 i 'm tough 
    0.50	 i 'm well 
    0.33	 i won . 
    0.17	 go calm 
    0.00	 slow down 
    ==  	 slow down . 	 	 	 ==
    
    3000 samples, 40 epochs, toy example: train data
    ==  	 it was long . 	 	 	 ==
    1.00	 it was long 
    0.83	 it was long 
    0.67	 it was new 
    0.50	 it was new 
    0.33	 it was wrong 
    0.17	 is that 
    0.00	 is that 
    ==  	 is that so ? 	 	 	 ==
    
    ==  	 i was ready . 	 	 	 ==
    1.00	 i was ready 
    0.83	 i was ready 
    0.67	 do n't die 
    0.50	 do n't die 
    0.33	 do n't lie 
    0.17	 he is here 
    0.00	 he is here 
    ==  	 he is here ! 	 	 	 ==
    
    ==  	 i feel cold . 	 	 	 ==
    1.00	 i feel cold 
    0.83	 i feel cold 
    0.67	 i feel . 
    0.50	 feel this 
    0.33	 bring wine 
    0.17	 say goodbye 
    0.00	 say goodbye 
    ==  	 say goodbye . 	 	 	 	 ==
```

