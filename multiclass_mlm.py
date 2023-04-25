from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs, ClassificationModel, ClassificationArgs
from example_based import example_based_accuracy, example_based_recall, example_based_precision, example_based_f1
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import argparse
from sklearn import metrics
import pandas as pd
import numpy as np
import csv
import os
import json

from transformers import AutoTokenizer, AutoModelForMaskedLM
from simpletransformers.classification import MLMForSequenceClassification


def report_per_epoch(args, test_df, seed, model_configs):
    list_of_results = []
    for epoch in range(1, args.epochs_per_seed+1):  
        end_dir = "epoch-"+str(epoch)
        dirs = [f for f in os.listdir(args.output_dir) if f[-len(end_dir):] == end_dir and os.path.isdir(os.path.join(args.output_dir, f))]
        if len(dirs) == 0:
            print("\nCheckpoint not found for epoch", str(epoch))       
        else:
            checkpoint_dir = os.path.join(args.output_dir, dirs[0])
            if args.task == "multiclass":

                # Create a MultiLabelClassificationModel
                model = ClassificationModel(model_configs["architecture"], checkpoint_dir)

                list_test_df = [str(i) for i in test_df['text'].values]

                # Evaluating the model on test data
                predictions, raw_outputs = model.predict(list_test_df)
                truth = list(test_df.labels)
                result_np, model_outputs, wrong_predictions = model.eval_model(test_df)

                # Collecting relevant results
                result = {k: float(v) for k, v in result_np.items()}

                result["acc"] = metrics.accuracy_score(truth, predictions)
                result["prec_micro"] = metrics.precision_score(truth, predictions, average='micro')
                result["prec_macro"] = metrics.precision_score(truth, predictions, average='macro')
                result["rec_micro"] = metrics.recall_score(truth, predictions, average='micro')
                result["rec_macro"] = metrics.recall_score(truth, predictions, average='macro')
                result["f1_micro"] = metrics.f1_score(truth, predictions, average='micro')
                result["f1_macro"] = metrics.f1_score(truth, predictions, average='macro')


            ## Other relevant information
            result["data_name"] = args.dataset
            result["model_name"] = model_configs["model_name"]
            result["seed"] = seed
            result["train_batch_size"] = args.train_batch_size
            result["epoch"] = epoch

            list_of_results.append(result)

    results_df = pd.DataFrame.from_dict(list_of_results, orient='columns')
    outfile_report = os.path.join("./logs/", str(args.dataset)+"_full_report.csv")

    if os.path.isfile(outfile_report):
        results_df.to_csv(outfile_report, mode='a', header=False, index=False)
    else:
        results_df.to_csv(outfile_report, mode='a', header=True, index=False)

def train_multiclass(args, train_df, eval_df, test_df, seed, model_configs):

    # Load the pre-trained MLM model and tokenizer
    print ("train_multiclass: model_configs["model_path"]", model_configs["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_configs["model_path"])
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_configs["model_path"])

    # Tokenize the training data using the MLM tokenizer
    train_text = train_df["text"].tolist()
    train_encodings = tokenizer(train_text, truncation=True, padding=True, return_tensors="pt")

    # Create a MLMForSequenceClassification model from the pre-trained model
    mlm_classifier = MLMForSequenceClassification.from_pretrained(model_configs["model_path"], num_labels=args.num_labels)

    # Fine-tune the MLM model on the training data
    mlm_classifier.train_model(train_encodings["input_ids"], train_encodings["attention_mask"], labels=train_df["labels"])


    # Training Arguments
    model_args = ClassificationArgs()
    model_args.manual_seed = seed
    model_args.best_model_dir = os.path.join(args.output_dir, "best_model", "")
    model_args.output_dir =  args.output_dir
    model_args.num_train_epochs = args.epochs_per_seed
    model_args.fp16 = False
    model_args.max_seq_length = args.max_seq_length
    model_args.train_batch_size = args.train_batch_size
    model_args.save_steps = -1
    model_args.use_multiprocessing = False
    # model_args.save_model_every_epoch = False
    if "do_lower_case" in model_configs:
        model_args.do_lower_case = model_configs["do_lower_case"]
    model_args.evaluate_during_training = True
    model_args.save_best_model = True
    model_args.save_eval_checkpoints = False
    # model_args.no_save = True
    model_args.overwrite_output_dir = True

    if not args.report_per_epoch:
        model_args.save_model_every_epoch = False
        model_args.no_save = True

    # Create a MultiLabelClassificationModel
    architecture = model_configs["architecture"]
    pretrained_model = mlm_classifier.model #model_configs["model_path"]
    model = ClassificationModel(architecture, pretrained_model, num_labels=args.num_labels, args=model_args)

    # Train the model
    model.train_model(train_df, eval_df=eval_df)
    list_test_df = [str(i) for i in test_df['text'].values]
    # Evaluating the model on test data
    predictions, raw_outputs = model.predict(list_test_df)
    truth = list(test_df.labels)
    result_np, model_outputs, wrong_predictions = model.eval_model(test_df)
    
    # Collecting relevant results
    result = {k: float(v) for k, v in result_np.items()}

    result["acc"] = metrics.accuracy_score(truth, predictions)
    result["prec_micro"] = metrics.precision_score(truth, predictions, average='micro')
    result["prec_macro"] = metrics.precision_score(truth, predictions, average='macro')
    result["rec_micro"] = metrics.recall_score(truth, predictions, average='micro')
    result["rec_macro"] = metrics.recall_score(truth, predictions, average='macro')
    result["f1_micro"] = metrics.f1_score(truth, predictions, average='micro')    
    result["f1_macro"] = metrics.f1_score(truth, predictions, average='macro')

    print ("PRINTING results", result)
    return result

def train_multi_seed(args, train_df, eval_df, test_df, model_configs):
    init_seed = args.initial_seed
    for curr_seed in range(init_seed, init_seed + args.num_of_seeds):
        
        if args.task == "multiclass":
            result = train_multiclass(args, train_df, eval_df, test_df, curr_seed, model_configs)
        
        # Recording best results for a given seed
        log_filename = os.path.join("./logs/", args.dataset+"_best_results.json")
        out_dict = {"data_name":args.dataset, "model_name": model_configs["model_name"], "seed": curr_seed, "train_batch_size": args.train_batch_size, "epochs": args.epochs_per_seed, "result": result}

        with open(log_filename, 'a') as fout:
            json.dump(out_dict, fout)
            fout.write('\n')
        
        if args.report_per_epoch:
            report_per_epoch(args, test_df, curr_seed, model_configs)

def loadData(args):
    if args.task == "multiclass":

        train_data = []
        max_label = 0
        with open(os.path.join(args.data_dir, "train.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = int(row[1:][0])
                max_label = max(max_label,labels)
                train_data.append([text, labels])
        train_df = pd.DataFrame(train_data, columns=['text', 'labels'])
        num_labels = max_label+1

        eval_data = []
        with open(os.path.join(args.data_dir, "dev.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = int(row[1:][0])
                eval_data.append([text, labels])
        eval_df = pd.DataFrame(eval_data, columns=['text', 'labels'])

        test_data = []
        with open(os.path.join(args.data_dir, "test.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = int(row[1:][0])
                test_data.append([text, labels])

        test_df = pd.DataFrame(test_data, columns=['text', 'labels'])

        return train_df, eval_df, test_df, num_labels

def main():

    parser = argparse.ArgumentParser()

    ## Main parameters
    parser.add_argument("--dataset",
                        default = "Tweets_mlm",
                        type=str,
                        help="The input dataset.")
    parser.add_argument("--report_per_epoch",
                        default = False,
                        action='store_true',
                        help="If true, will output the report per epoch.")
    args = parser.parse_args()

    ## Loading the configurations and relevant variables
    data_json = os.path.join("./configs/", args.dataset+".json")
    with open(data_json, 'r') as fp:
        data_configs = json.load(fp)

    for k,v in data_configs.items():
        setattr(args, k, v)

    args.data_dir = os.path.join("./data/", args.dataset, "")

    ## Loading the datasets
    train_df, eval_df, test_df, args.num_labels = loadData(args)
   
    ## Running experiments for all the models in configs:
    for model_configs in args.models:
        
        # args.output_dir = os.path.join("./outputs/", args.dataset + "_" + model_configs["model_name"], "")
        args.output_dir = os.path.join("./outputs/", args.dataset, "")

        train_multi_seed(args, train_df, eval_df, test_df, model_configs)
        
if __name__ == "__main__":
    main()