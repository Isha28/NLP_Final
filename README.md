# Twitter Hate Speech Detection with ConfliBert

git clone https://github.com/Isha28/NLP_Final.git
cd 'NLP_Final'


## Environment setup
The below requirements mentioned in setup.sh must be installed:
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install transformers==4.17.0
pip install numpy==1.19.2 
pip install scikit-learn==0.24.2
pip install simpletransformers
pip install pandas

Run the setup file:
```
bash setup.sh 
```


## To collect Tweets from Twitter using API:
Add Twitter API credentials in the get_tweets.py file for below. 
<ol>
  <li>consumer_key = "..."</li>
  <li>consumer_secret = "..."</li>
  <li>access_token = "..."</li>
  <li>access_token_secret = "..."</li>
</ol>

After updating these details, run get_tweets file to collect tweets by giving appropriate search_words in file.
```
python get_tweets.py
```

We collected Tweets manually from Twitter for political violence and hate speech classes as the authorization to use API was restricted from Twitter.


## To preprocess tweets 
Run the file tweet_preprocess where input file is the collected and annotated input tweets dataset containing political violence/hate speech. The output files are preprocessed and split train, test and dev files.


## To pretrain ConfliBERT for Masked Language Modeling
The pretrain_mlm file below takes unlabeled tweets data as input to pretrain.
```
CUDA_VISIBLE_DEVICES=0 python pretrain_mlm.py
```

This will save the output bin files of four versions of pretrained ConfliBERT MLM models to local disk. We have uploaded the output bin, config and vocab files in huggingface to use them in performance evaluation of classification tasks.

<ol>
  <li>https://huggingface.co/ipadmanaban/Masked-2KTweets-ConfliBERT-scr-cased/tree/main</li>
  <li>https://huggingface.co/ipadmanaban/Masked-2KTweets-ConfliBERT-scr-uncased/tree/main</li>
  <li>https://huggingface.co/ipadmanaban/Masked-2KTweets-ConfliBERT-cont-uncased/tree/main</li>
  <li>https://huggingface.co/ipadmanaban/Masked-2KTweets-ConfliBERT-cont-cased/tree/main</li>
</ol>


## To run the classification tasks, we need to prepare two directories in data and configs folder
Create a new directory in data folder with dataset name containing three files train.tsv, test.tsv and dev.tsv obtained as a result of executing tweet_preprocess script.
Create a new json file with the same dataset name as above in configs folder. The sample config file should be structured as below:

```
{
    "task": "binary", - Here, binary or multiclass classification should be specified
    "num_of_seeds": 1, - Specify number of speeds
    "initial_seed": 123,
    "epochs_per_seed": 5, - Specify number of epochs 
    "train_batch_size": 8, 
    "max_seq_length": 128,
    "models": [
        {
            "model_name": "Masked-2KTweets-ConfliBERT-scr-cased", - Specify the model name as mentioned in huggingface
            "model_path": "ipadmanaban/Masked-2KTweets-ConfliBERT-scr-cased", - Specify the entire model path
            "architecture": "bert",
            "do_lower_case": false
        },
        {
            "model_name": "ConfliBERT-scr-cased",
            "model_path": "snowood1/ConfliBERT-scr-cased",
            "architecture": "bert",
            "do_lower_case": false
        },
        {
            "model_name": "bert-base-cased",
            "model_path": "bert-base-cased",
            "architecture": "bert",
            "do_lower_case": false
        }
    ]
}
```

The above is a sample with one flavor of BERT, ConfliBERT and proposed ConfliBERT-MLM. To run classification task, dataset directory name must be specified in --dataset. 

```
CUDA_VISIBLE_DEVICES=0 python finetune_data.py --dataset Tweets_proc --report_per_epoch
```

The results are redirected to logs folder with dataset name. The results will have two set of files - best results json file highlighting performance metrics such as accuracy, f1 score, recall, precision of models that are best comparatively. Another file is full report csv file which has performance metrics for every epoch for every model.


## Citation

```
@inproceedings{hu2022conflibert,
    title={ConfliBERT: A Pre-trained Language Model for Political Conflict and Violence},
    author={Hu, Yibo and Hosseini, MohammadSaleh and Parolin, Erick Skorupa and Osorio, Javier and Khan, Latifur and Brandt, Patrick and D’Orazio, Vito},
    booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
    pages={5469--5482},
    year={2022}
}
```