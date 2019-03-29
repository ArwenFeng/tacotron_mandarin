# Tacotron for mandarin

An implementation of Tacotron speech synthesis in TensorFlow for mandarin. An sythesized audio sample is given in tmp\ directory.


## Background

In April 2017, Google published a paper, [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf),
where they present a neural text-to-speech model that learns to synthesize speech directly from
(text, audio) pairs. However, they didn't release their source code or training data. Some time later, Keithito made an independent attempt to provide an open-source implementation of the model described in their paper.

Recently, I forked it from Keithito and trained the model on an opensource mandarin dataset. The result sounds good. Hopefully, it can do better with more effort.



## Quick Start

### Installing dependencies

1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better
   performance, install with GPU support if it's available. This code works with TensorFlow 1.3 and later.

3. Install requirements:
The packages in requirements.txt may not be all necessary, I just freezed my workspace and got tired of cleaning it. 
   ```
   pip install -r requirements.txt
   ```


### Using a pre-trained model

1. **Download and unpack a model**:
 url: https://pan.baidu.com/s/1xekQvt7BgUlUuuZnJzVlZA 
password：efkz 
Or you can scan the following QR code.
![79098607b50113607150c7796cc30231.png](en-resource://database/627:1)


2. **Run the demo server**:
Note that: When you run the demo server, you should use chinese_cleaners. Remember to confirm it in [hparams.py](hparams.py).
    ```
    cleaners='chinese_cleaners',
    ```
    After that, you can run the demo server!
   ```
   python3 demo_server.py --checkpoint tmp/model.ckpt-64000
   ```

3. **Point your browser at localhost:9000**
   * Type what you want to synthesize

4. **Here are some input samples.**
    * 今天天气真好
    * 孙杨拿了{guan4}军
5. **Attention:**
    * Do not input any punctuations.
    * Make sure to change numbers into chinese characters
    * When it comes to polyphone problem, you can use  pinyin like the example listed above.
    
### Training

*Note: you need at least 40GB of free disk space to train a model.*

1. **Download a speech dataset.**

    * [BZNSYP](
https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar) (from Leo Ma)


2. **Unpack the dataset into `~/tacotron_mandarin`**

   After unpacking, your tree should look like this for BZNSYP:
   ```
   tacotron_mandarin
     |- BZNSYP
         |- PhoneLabeling
         |- ProsodyLabeling
         |- Wave
   ```

   

3. **Preprocess the data**
   ```
   python3 preprocess.py --dataset BZNSYP
   ```
4. **Train a model**
Note that: When training BZNSYP dataset, you should use basic_cleaners. Remember to confirm it in [hparams.py](hparams.py)
    ```
    cleaners='basic_cleaners',
    ```
    After adjust hyperparameters, you can train your own models.
   ```
   python3 train.py
   ```
   Tunable hyperparameters are found in [hparams.py](hparams.py). You can adjust these at the command
   line using the `--hparams` flag, for example `--hparams="batch_size=16,outputs_per_step=2"`.
   The hyperparameters other than cleaners should generally be set to the same values at both training and eval time.
   
   The default hyperparameters are recommended for LJ Speech and other English-language data from Keithito. Although, I didn't change them, they work quite well on this BZNSYP mandarin dataset.

5. **Synthesize from a checkpoint**
   ```
   python3 demo_server.py --checkpoint ~/tacotron_mandarin/logs-tacotron/model.ckpt-64000
   ```
   Replace "64000" with the checkpoint number that you want to use, then open a browser
   to `localhost:9000` and type what you want to speak. Alternately, you can
   run [eval.py](eval.py) at the command line:
   ```
   python3 eval.py --checkpoint ~/tacotron_mandarin/logs-tacotron/model.ckpt-64000
   ```
   If you set the `--hparams` flag when training, set the same value here.


## Express thanks
**Origin Implementation**
  * By keithito: 
  https://github.com/keithito/tacotron
  
