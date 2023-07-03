from datasets import load_dataset
from tqdm.std import tqdm as std_tqdm
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import set_seed
from transformers import ProgressCallback
from transformers.trainer_utils import has_length
import argparse
import evaluate
import numpy as np

class tqdm(std_tqdm):
    def display(self, msg=None, pos=None):
        print(self.format_meter(**self.format_dict))

def print_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_train = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Total        : {total:,d}')
    print(f'Trainable    : {trainable:,d}')
    print(f'Non-Trainable: {non_train:,d}')

class CustomProgressCallback(ProgressCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar = tqdm(total=state.max_steps)
        self.current_step = 0

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_local_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(total=len(eval_dataloader), leave=self.training_bar is None)
            self.prediction_bar.update(1)
            try:
                self.prediction_bar.set_description(f'epoch: {state.epoch} - best_metric: {state.best_metric or -1.0:.4f}')
            except:
                pass


def build_preprocessor(tokenizer, max_input_length, prefix):
    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["func"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)


        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            # 2 tokens is enough for the 'true' or 'false'
            text_labels = [ 'true' if e == 1 else 'false' for e in examples['target'] ]
            labels = tokenizer(text_labels, max_length=2, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    return preprocess_function

def build_compute_metrics(tokenizer, metrics):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = np.array(decoded_preds) == 'true'
        decoded_labels = np.array(decoded_labels) == 'true'
        
        result = {'preds': decoded_preds.astype(int).tolist() }
        for metric in metrics:
            result.update(metric.compute(
                predictions=decoded_preds,
                references=decoded_labels))

        return result
    return compute_metrics

def main(args):
    set_seed(args.seed)

    print('-' * 20)
    print('Model:', args.model_name_or_path)
    print('-' * 20)

    raw_datasets = load_dataset('json',
        data_files={
            'train': args.train_data_file,
            'validation': args.eval_data_file,
            'test': args.test_data_file,
        })

    metrics = [ evaluate.load(k) for k in [ 'accuracy', 'precision', 'recall', 'f1' ] ]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    preprocess_function = build_preprocessor(
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        prefix='defect_detection: ',
    )
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    print_params(model)

    trainer = Seq2SeqTrainer(
        model,
        Seq2SeqTrainingArguments(
            args.output_dir,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=args.epoch,
            predict_with_generate=True,
            fp16=args.fp16,
            push_to_hub=False,
            seed=args.seed,
            disable_tqdm=True,
        ),
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer, metrics),
        callbacks=[CustomProgressCallback()],
    )

    if args.do_train:
        trainer.train()
        trainer.save_model(args.output_dir + '/final')

    if args.do_eval:
        print('-' * 100)
        print('Validation:')
        result = trainer.evaluate(tokenized_datasets['validation'])
        for k, v in result.items():
            if 'pred' in k: continue
            print(k, ':', round(v, 4))
    
    # FIXME Change to test function???
    if args.do_test:
        print('-' * 100)
        print('Test')

        result = trainer.evaluate(tokenized_datasets['test'])
        for k, v in result.items():
            if 'pred' in k: continue
            print(k, ':', round(v, 4))
        
        if args.predictions:
            with open(args.predictions, 'w') as f:
                indices = [ x['idx'] for x in tokenized_datasets['test']]
                for idx, pred in zip(indices, result['eval_preds']):
                    f.write(f"{idx}\t{pred}\n")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name_or_path", default='razent/cotext-1-ccg', type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--tokenizer_name", default='razent/cotext-1-ccg', type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--max_input_length", default=1024, type=int,
                        help="Max input length after tokenization")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    # parser.add_argument("--evaluate_during_training", action='store_true',
    #                     help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    #                     help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--weight_decay", default=0.0, type=float,
    #                     help="Weight deay if we apply some.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    # parser.add_argument("--num_train_epochs", default=1.0, type=float,
    #                     help="Total number of training epochs to perform.")
    # parser.add_argument("--max_steps", default=-1, type=int,e
    #                     help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    # parser.add_argument("--warmup_steps", default=0, type=int,
    #                     help="Linear warmup over warmup_steps.")

    # parser.add_argument('--logging_steps', type=int, default=50,
    #                     help="Log every X updates steps.")
    # parser.add_argument('--save_steps', type=int, default=50,
    #                     help="Save checkpoint every X updates steps.")
    # parser.add_argument('--save_total_limit', type=int, default=None,
    #                     help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    # parser.add_argument("--eval_all_checkpoints", action='store_true',
    #                     help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    # parser.add_argument("--no_cuda", action='store_true',
    #                     help="Avoid using CUDA when available")
    # parser.add_argument('--overwrite_output_dir', action='store_true',
    #                     help="Overwrite the content of the output directory")
    # parser.add_argument('--overwrite_cache', action='store_true',
    #                     help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="Number of epochs")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    # parser.add_argument('--fp16_opt_level', type=str, default='O1',
    #                     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    #                          "See details at https://nvidia.github.io/apex/amp.html")
    # parser.add_argument("--local_rank", type=int, default=-1,
    #                     help="For distributed training: local_rank")
    # parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    # parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--predictions', type=str, default=None)
    args = parser.parse_args()
    main(args)
