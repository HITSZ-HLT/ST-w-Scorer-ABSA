import os 
import time
import random

import torch
import lightning as pl 
pl.seed_everything(42)
from lightning.pytorch.callbacks import BasePredictionWriter

from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup
from transformers import T5ForConditionalGeneration
from transformers import AutoConfig, AutoTokenizer

from utils import params_count, load_json, tokenize, load_line_json, save_line_json, save_json, tgenerate_batch, simple_text_len, auto_init
from utils.quad import make_quads_seq, parse_quads_seq, get_quad_aspect_opinion_num
from utils.quad_result import Result



class DataModule(pl.LightningDataModule):
    @auto_init
    def __init__(
        self,
        model_name_or_path: str='',
        max_seq_length: int = -1,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        seed: int = 42,
        data_dir: str = '',
        dataset: str = '',
        mode: str = 'train_test', # ('train_test', 'predict')
        self_training_data_dir: str = '',
        filter_setting: str = 'none',
    ):

        super().__init__()

        self.data_dir = self.data_dir if self.dataset == '' else os.path.join(self.data_dir, self.dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        print(f'data-dir: {self.data_dir}')

    def load_labeled_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        dev_file_name   = os.path.join(self.data_dir, 'dev.json')
        test_file_name  = os.path.join(self.data_dir, 'test.json')

        train_examples = load_json(train_file_name)
        dev_examples   = load_json(dev_file_name)
        test_examples  = load_json(test_file_name)

        self.raw_datasets = {
            'train': train_examples, 
            'dev'  : dev_examples,
            'test' : test_examples
        }

        if self.self_training_data_dir and self.filter_setting != 'none':
            self.add_self_training_data()

        print('-----------data statistic-------------')
        for mode in ('train', 'dev', 'test'):
            num_sentences = len(self.raw_datasets[mode])
            num_quads     = sum([get_quad_aspect_opinion_num(example)[0] for example in self.raw_datasets[mode]])
            print(f'{mode.upper():<5} | Sentences: {num_sentences:<5} | Quad: {num_quads:<5}')

        print('--------------------------------------')

    def add_self_training_data(self):
        setting, k = self.filter_setting.split('_')
        k = int(k)

        print(f'setting: {setting} | k: {k}')
        try:
            self_training_data = load_json(self.self_training_data_dir)
        except:
            self_training_data = list(load_line_json(self.self_training_data_dir))

        if len(self_training_data) >= 110_000:
            self_training_data = self_training_data[10_000:110_000]

        self_training_data = [{
            'ID': example['ID'],
            'sentence': example['sentence'],
            'quads_seq': example['quad_preds'][0],
            'reward': example['reward'] if 'reward' in example else [None],
            'quad_preds': example['quad_preds'],
        } for example in self_training_data]

        print(len(self_training_data))

        if setting == 'full':
            pass

        else:
            try:
                start, end = setting.split('-')
                start = int(start) / 100 * len(self_training_data)
                end   = int(end) / 100 * len(self_training_data)

                self_training_data = sorted(self_training_data, key=lambda e: e['reward'][0], reverse=True)[int(start): int(end)]
            
            except:
                raise NotImplementedError(f'Unknown setting: {self.filter_setting}')

        if k > 0:
            random.seed(self.seed)
            self_training_data = random.sample(self_training_data, k=k)
            mean_reward = sum(example['reward'][0] for example in self_training_data) / len(self_training_data)
            print(f'mean_reward: {mean_reward}')
            self.raw_datasets['train'] += self_training_data

    def load_unlabeled_dataset(self, max_example_num=1_000_000):
        import spacy 
        import re

        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('sentencizer')

        min_length = 5
        max_length = 100

        dataset = list(load_line_json(self.data_dir))

        def process_and_filter(sentence):
            sentence = str(sentence).strip()
            sentence = sentence.replace('\r', '')
            # '(good)' -> '( good )'
            sentence = re.sub(r'\((?P<v1>[^ ])(?P<v2>.*)(?P<v3>[^ ])\)', lambda x: '( ' + x.group('v1') + x.group('v2') + x.group('v3') + ' )', sentence)

            if not (min_length <= simple_text_len(sentence) <= max_length):
                return None

            return sentence            

        predict_examples = []
        for batch_examples in tgenerate_batch(dataset, bz=32):

            texts = [example['Text'] for example in batch_examples]
            docs  = nlp.pipe(texts, disable=['tagger', 'tok2vec', 'parser', 'lemmatizer', 'ner'])

            for doc, example in zip(docs, batch_examples):
                for i, sentence in enumerate(doc.sents):
                    if (sentence := process_and_filter(sentence)) is not None:
                        new_example = {
                            'ID': f"{example['ID']}-{i}",
                            'sentence': sentence,
                            'full_review': example['Text']
                        }
                        predict_examples.append(new_example)

                if max_example_num > 0 and len(predict_examples) >= max_example_num:
                    break

        self.raw_datasets = {'predict': predict_examples}

        print('-----------data statistic-------------')
        print('Predict', len(self.raw_datasets['predict']))

    def prepare_data(self):
        if self.mode == 'train_test':
            self.load_labeled_dataset()

        elif self.mode == 'predict':
            self.load_unlabeled_dataset()

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(
            dataset=self.raw_datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            prefetch_factor=8,
            num_workers=1,
            collate_fn=DataCollator(
                tokenizer=self.tokenizer, 
                max_seq_length=self.max_seq_length,
                mode=mode,
                dataset=self.dataset
            )
        )

        print('dataloader-'+mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False)

    def predict_dataloader(self):
        return self.get_dataloader("predict", self.eval_batch_size, shuffle=False)



class DataCollator:
    @auto_init
    def __init__(self, tokenizer, max_seq_length, mode, dataset):
        pass

    def tok(self, text, max_seq_length):
        return tokenize(self.tokenizer, text, max_seq_length)
    
    def __call__(self, examples):
        text = [example['sentence'] for example in examples]
        batch_encodings = self.tok(text, -1)

        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']

        labels = None
        if self.mode in ('train', 'dev', 'test'):
            labels = self.make_labels(examples)

        if self.max_seq_length > 0:
            input_ids = input_ids[:, :self.max_seq_length]
            attention_mask = attention_mask[:, :self.max_seq_length]
            if labels is not None:
                labels = labels[:, :self.max_seq_length]

        return {
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'labels'        : labels,
            'examples'      : examples
        }

    def make_labels(self, examples):
        target_seqs = [
            self.make_quads_seq(example)
            for example in examples
        ]

        batch_encodings = self.tok(target_seqs, -1)
        labels = batch_encodings['input_ids']
        labels = torch.tensor([
            [(l if l != self.tokenizer.pad_token_id else -100)
             for l in label]
            for label in labels
        ])

        return labels

    def make_quads_seq(self, example):
        if 'quads_seq' in example:
            return example['quads_seq']

        return make_quads_seq(example)



class LightningModule(pl.LightningModule):
    @auto_init
    def __init__(
        self, 
        learning_rate: float=2e-4,
        adam_epsilon: float=1e-6,
        weight_decay: float=0.,
        warmup_steps: int=0,
        seed: int=42,
        dataset: str='',
        output_dir: str='',
        subname: str='',
        model_name_or_path: str='',
        min_con_thre=0.7,
        avg_con_thre=0.9,
    ):

        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name_or_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        print('---------------------------------------------')
        print(self.model_name_or_path)
        print('total params_count:', params_count(self.model))
        # print(self.model.config)
        print('---------------------------------------------')

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def save_model(self):
        dir_name = os.path.join(self.output_dir, 'model', f'dataset={self.dataset},b={self.subname},seed={self.seed}')
        print(f'## save model to {dir_name}')
        self.model.config.time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        self.model.save_pretrained(dir_name)
        self.tokenizer.save_pretrained(dir_name)

    def forward(self, **inputs):
        examples = inputs.pop('examples')
        output   = self.model(**inputs)
        return {'loss': output[0]}

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def eval_step(self, batch, batch_idx, num_beams=1):
        generated_ids = self.model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=100,
            num_beams=num_beams,
            num_return_sequences=num_beams,
        )
        generateds = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        if num_beams > 1:
            generateds = [generateds[i:i+num_beams] for i in range(0, len(generateds), num_beams)]

        return {
            'examples': batch['examples'],
            'predictions': generateds
        }

    def validation_step(self, batch, batch_idx):
        output = self.eval_step(batch, batch_idx)
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self):
        self.current_val_result = Result.parse_from(self.validation_step_outputs)
        self.current_val_result.cal_metric()
    
        self.update_result = False
        if (not hasattr(self, 'best_val_result')) or (self.best_val_result < self.current_val_result):
            self.best_val_result = self.current_val_result
            self.update_result = True
            # select model by devlopment set
            self.save_model()

        self.validation_step_outputs.clear()

    # def on_train_end(self):
    #     self.save_model()

    def test_step(self, batch, batch_idx):
        output = self.eval_step(batch, batch_idx, num_beams=4)
        self.test_step_outputs.append(output)

    def on_test_epoch_end(self):
        self.test_result = Result.parse_from(self.test_step_outputs)
        self.test_result.cal_metric()
        self.test_result.save_metric(
            self.output_dir, 
            self.model_name_or_path, 
            self.subname, 
            self.dataset, 
            self.seed,
            self.learning_rate,
        )
        self.test_result.save_prediction(
            self.output_dir, 
            self.model_name_or_path, 
            self.subname, 
            self.dataset, 
            self.seed,
            self.learning_rate,
        )
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):

            generated_outputs = self.model.generate(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=100,
                num_beams=1,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
            )

            min_con, avg_con = self.get_confidence(generated_outputs)

            index = (min_con>self.min_con_thre) * (avg_con>self.avg_con_thre)

            input_ids = batch['input_ids'][index]
            attention_mask = batch['attention_mask'][index]

            examples = batch['examples']
            examples = [examples[i] for i in range(len(examples)) if index[i]]

            min_con = min_con[index]
            avg_con = avg_con[index]

            num_beams = 4
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=100,
                num_return_sequences=num_beams,
                num_beams=num_beams,
            )

            generateds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            generated_beams = [generateds[i:i+num_beams] for i in range(0, len(generateds), num_beams)]

        return {
            'examples': examples,
            'predictions': generated_beams,
            'min_con': min_con,
            'avg_con': avg_con,
        }

    def get_confidence(self, generated_outputs):
        input_ids = generated_outputs['sequences']
        attention_mask = self.get_mask(input_ids)[:, 1:] # 1: to remove decoder_start_id

        probs = torch.stack(generated_outputs.scores, dim=1)
        probs = F.log_softmax(probs, dim=-1)
        confidence = probs.max(dim=-1)[0]

        confidence[~attention_mask.bool()] = 0
        min_confidence = confidence.min(dim=-1)[0].exp().detach().cpu().numpy()

        avg_confidence = confidence.sum(dim=-1) / attention_mask.sum(dim=-1)
        avg_confidence = avg_confidence.exp().detach().cpu().numpy()

        return min_confidence, avg_confidence

    def get_mask(self, input_ids):
        eos_token_id = self.model.config.eos_token_id
        pad_token_id = self.model.config.pad_token_id

        eos_flag = (input_ids == eos_token_id)
        eos_flag = torch.cat([eos_flag[:, :1], eos_flag[:, :-1]], dim=1)
        attention_mask = torch.cumsum(eos_flag, dim=1)
        attention_mask = (attention_mask == 0).bool()

        return attention_mask.long()

    def configure_optimizers(self):

        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon, 
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]



class CustomWriter(BasePredictionWriter):
    def __init__(self, argument_parser, name_space, write_interval='epoch'):
        super().__init__(write_interval)
        self.argument_parser = argument_parser
        self.name_space = name_space
        self.output_dir = name_space.model.output_dir

    def on_validation_end(self, trainer, pl_module):
        if not pl_module.update_result:
            return 

        if hasattr(pl_module, 'current_train_result'):
            pl_module.current_train_result.report()
        print('------------------------------------------------------------')
        print('[current]', end=' ')
        pl_module.current_val_result.report()

        print('[best]   ', end=' ')
        pl_module.best_val_result.report()
        print('------------------------------------------------------------\n')

    def on_test_end(self, trainer, pl_module):
        pl_module.test_result.report()

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):

        output_examples = []
        N = 0
        for output in tqdm(predictions):
            examples = output['examples']
            predictions = output['predictions']
            min_confidence = output['min_con']
            avg_confidence = output['avg_con']

            for example, prediction, min_con, avg_con in zip(examples, predictions, min_confidence, avg_confidence):

                    output_examples.append({
                        'ID': example['ID'],
                        'sentence': example['sentence'],
                        'quad_preds': prediction,
                        'quads_seq' : example.get('quads_seq'),
                        'min_con' : float(min_con),
                        'avg_con' : float(avg_con),
                        'full_review': example['full_review'],
                    })

        print(f'save {len(output_examples)} to', self.output_dir)
        if len(output_examples) > 10_000:
            save_line_json(output_examples, self.output_dir)
        else:
            save_json(output_examples, self.output_dir)


def cli_main():
    from lightning.pytorch.cli import LightningCLI
    cli = LightningCLI(LightningModule, DataModule, CustomWriter)


if __name__ == '__main__':
    cli_main()
