# 用Tacotron做普通话合成

用TensorFlow实现的普通话语音合成。一个合成的音频样例放在了tmp文件夹下。


## 背景

2017年4月, Google 发了一篇文章, [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf),
提出了一种端到端的语音合成模型，可直接以（文本，语音）对训练。然而，他们没有提供源代码。不久后， Keithito 在github上开源了他根据文章所作的tensorflow实现。最近我阅读了这份代码，并将其略作修改，用于中文的语音合成。合成的结果听起来结果不错。但也存在一些问题，比如没有考虑标点符号，数字等等。

## 快速上手

### 安装依赖库

1. 安装Python3。

2. 安装最新版的 [TensorFlow](https://www.tensorflow.org/install/) 。用于训练的话，最好是GPU版本。

3. 安装必要的库:
在requirements.txt中的库不一定是必须的。我只是把我的虚拟环境里的库都列出来了，但没有仔细分辨哪些是一定要，哪些不一定要。建议还是根据自己的环境手动安装报错没有的库。. 
   ```
   pip install -r requirements.txt
   ```


### 使用一个训练好的模型

1. **下载解压模型**

    [百度网盘](https://pan.baidu.com/s/1xekQvt7BgUlUuuZnJzVlZA) 密码：efkz 
    
2. **运行demo_server.py**:
* 注意: 当你运行[demo_server.py](demo_server.py)时， 你应当使用chinese_cleaners。记得在[hparams.py](hparams.py)中确认一下使用的是chinese_cleaners。
    ```
    cleaners='chinese_cleaners',
    ```
    然后你就可以运行[demo_server.py](demo_server.py)了!
   ```
   python3 demo_server.py --checkpoint tmp/model.ckpt-64000
   ```

3. **在你的浏览器中输入网址 localhost:9000**
   * 输入你想合成的语句
        * 今天天气真好
        * 孙杨拿了{guan4}军
 4. **注意**
     * 不要输入标点符号，因为标点没有参与训练。
     * 如果想要输入数字，要用中文输入比如：十一，幺三九。
     * 如果输入的字符是多音字，你可能需要用拼音指定读音，正如上述第二个例子所示。
    
### 训练

*注意: 你需要至少40G的硬盘空间来训练模型。*

1. **下载中文语音数据集**

    * [BZNSYP](
https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar) ( [Leo Ma](https://github.com/begeekmyfriend?tab=repositories)公开的数据集)


2. **解压数据集到当前目录下 `~/tacotron_mandarin`**

   解压后BZNSYP的文件目录应该是这样的:
   ```
   tacotron_mandarin
     |- BZNSYP
         |- PhoneLabeling
         |- ProsodyLabeling
         |- Wave
   ```

3. **预处理数据**
   ```
   python3 preprocess.py --dataset BZNSYP
   ```

4. **训练模型**
注意: 当用BZNSYP数据集训练时，你应当使用basic_cleaners。记得在[hparams.py](hparams.py)中确认这一点。
    ```
    cleaners='basic_cleaners',
    ```
    调整好你想要的参数后，你就可以训练了。
   ```
   python3 train.py
   ```


   在 [hparams.py](hparams.py)中列出了可改变的参数。你可以直接修改文件，或者用类似`--hparams="batch_size=16,outputs_per_step=2"`的命令修改它们。除了cleaners参数在处理中文和拼音时需要不同，其他参数在训练和评测时应该保持一致。
   
   默认参数是原作者Keithito训练LJSpeech数据集时用的。在训练这份中文数据集时我并没有改变它们，合成的效果已经不错了。

5. **用训练时产生的参数文件合成语音**
   ```
   python3 demo_server.py --checkpoint ~/tacotron_mandarin/logs-tacotron/model.ckpt-64000
   ```
   你可以使用其他步数替换"64000" , 然后打开`localhost:9000` 输入你想合成的内容。或者, 你可以在命令行运行[eval.py](eval.py) :
   ```
   python3 eval.py --checkpoint ~/tacotron_mandarin/logs-tacotron/model.ckpt-64000
   ```

## 致谢
**源代码来自keithito** 
  https://github.com/keithito/tacotron
  
