import os
import json
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

from utils import params_count, load_json, tokenize, load_line_json, append_new_line, save_json, auto_init
from utils.quad import make_quads_seq, get_quad_aspect_opinion_num, parse_quads_seq

from sklearn.metrics import f1_score, accuracy_score




class RewardModel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def get_rewards(self, input_ids, attention_mask, decoder_labels):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=decoder_labels
        )

        lm_logits = outputs.logits

        decoder_labels[decoder_labels==-100] = 0
        logprobs = logprobs_of_labels(lm_logits, decoder_labels)
        mask = (decoder_labels!=0).to(logprobs.dtype)

        rewards = (logprobs * mask).sum(dim=-1)
        # mean_rewards = (logprobs * mask).sum(dim=-1) / mask.sum(dim=-1)

        return rewards


def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels
    These are calculated from the logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)



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
        preference_data_dir: str='',
        setting: str = '01234+',
        k: int = 1000,
        use_ai_preference: bool = True,
        mode: str = 'train_test',
    ):

        super().__init__()

        self.data_dir = self.data_dir if self.dataset == '' else os.path.join(self.data_dir, self.dataset)
        self.preference_data_dir = self.preference_data_dir if self.dataset == '' else os.path.join(self.preference_data_dir, self.dataset)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        print(f'data-dir: {self.data_dir}')
        print(f'preference_data_dir: {self.preference_data_dir}')

    def load_dataset(self):
        train_examples = load_json(os.path.join(self.data_dir, 'train.json'))
        test_comp_examples = load_json(os.path.join(self.preference_data_dir, 'test.json'))

        if not self.use_ai_preference:
            train_comp_examples = load_json(os.path.join(self.preference_data_dir, 'train.json'))
        else:
            train_comp_examples = load_json(os.path.join(self.preference_data_dir, 'ai_train.json'))

        self.stat(train_comp_examples)
        self.stat(test_comp_examples)

        def sample_k(comp_examples):
            if self.k >= len(comp_examples):
                return comp_examples
            elif self.k > 0:
                random.shuffle(comp_examples)
                return comp_examples[:self.k]
            else:
                raise NotImplementedError(f'Incorrect k={self.k}.')

        def append_train_examples(comp_examples):
            N = len(train_examples) // 4
            for i in range(N):
                batch = train_examples[i*4 : i*4 + 4]
                example = {
                    'sentences': [x['sentence'] for x in batch],
                    'quad_preds': [make_quads_seq(x) for x in batch],
                    'perfered_option': 5
                }
                comp_examples.append(example)

        if self.setting == '0123':
            train_comp_examples = [example for example in train_comp_examples if (example['perfered_option'] in (0,1,2,3))]
            train_comp_examples = sample_k(train_comp_examples)
        
        elif self.setting == '01234':
            train_comp_examples = [example for example in train_comp_examples if (example['perfered_option'] in (0,1,2,3,4))]
            train_comp_examples = sample_k(train_comp_examples)
        
        elif self.setting == '0123+':
            train_comp_examples = [example for example in train_comp_examples if (example['perfered_option'] in (0,1,2,3))]
            train_comp_examples = sample_k(train_comp_examples)
            append_train_examples(train_comp_examples)
        
        elif self.setting == '01234+':
            train_comp_examples = [example for example in train_comp_examples if (example['perfered_option'] in (0,1,2,3,4))]
            train_comp_examples = sample_k(train_comp_examples)
            append_train_examples(train_comp_examples)

        self.raw_datasets = {
            'train': train_comp_examples, 
            'dev'  : test_comp_examples,
            'test' : test_comp_examples
        }

        print('-----------data statistic-------------')
        print('train:', len(self.raw_datasets['train']))
        print('dev:  ', len(self.raw_datasets['dev']))
        print('test: ', len(self.raw_datasets['test']))
        print('--------------------------------------')

    def stat(self, comp_examples):
        key = 'perfered_option'
        a0 = sum([(example[key] == 0) for example in comp_examples if key in example])
        a1 = sum([(example[key] == 1) for example in comp_examples if key in example])
        a2 = sum([(example[key] == 2) for example in comp_examples if key in example])
        a3 = sum([(example[key] == 3) for example in comp_examples if key in example])
        a4 = sum([(example[key] == 4) for example in comp_examples if key in example])
        a5 = sum([(example[key] == 5) for example in comp_examples if key in example])
        a6 = sum([(example[key] == 6) for example in comp_examples if key in example])
        a7 = sum([(example[key] == 7) for example in comp_examples if key in example])
        a8 = sum([(example[key] == 8) for example in comp_examples if key in example])

        su = a0+a1+a2+a3+a4+a5+a6+a7+a8

        print(f'statistic({su}): ')
        print(f' - candidate {a0:03d}, {a1:03d}, {a2:03d}, {a3:03d}, total {a0+a1+a2+a3:03d} | correct {a4:03d} | unrelated {a5:03d} | no-sentiment {a6:03d} | 7 {a7:03d} | 8 {a8:03d}')
        print(f' - candidate {a0/su*100:02.0f}%, {a1/su*100:02.0f}%, {a2/su*100:02.0f}%, {a3/su*100:02.0f}%, total {(a0+a1+a2+a3)/su*100:02.0f}% | correct {a4/su*100:02.0f}% | unrelated {a5/su*100:02.0f}% | no-sentiment {a6/su*100:02.0f}% | 7 {a7/su*100:02.0f}% | 8 {a8/su*100:02.0f}%')

    def load_filter_dataset(self):
        data_dir = self.data_dir
        try:
            examples = load_json(data_dir)
        except:
            examples = list(load_line_json(data_dir))
            if len(examples) > 110000:
                examples = examples[10_000: 110_000]

        self.raw_datasets = {'predict': examples}
        print('-----------data statistic-------------')
        print('predict:', len(self.raw_datasets['predict']))
        print('--------------------------------------')

    def load_rerank_dataset(self):
        data_dir = self.data_dir
        examples = list(load_json(data_dir)['data'].values())

        for example in examples:
            example['quad_preds'] = example['prediction']

        self.raw_datasets = {'predict': examples}
        print('-----------data statistic-------------')
        print('predict:', len(self.raw_datasets['predict']))
        print('--------------------------------------')        

    def prepare_data(self):
        if self.mode == 'train_test':
            self.load_dataset()

        elif self.mode == 'filter':
            self.load_filter_dataset()

        elif self.mode == 'rerank':
            self.load_rerank_dataset()

    def get_dataloader(self, mode, batch_size, shuffle, drop_last=False):
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
                mode=self.mode,
            ),
            drop_last=drop_last,
        )

        print('dataloader-'+mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False, drop_last=False)

    def predict_dataloader(self):
        return self.get_dataloader("predict", self.eval_batch_size, shuffle=False, drop_last=False)



class DataCollator:
    @auto_init
    def __init__(self, tokenizer, max_seq_length, mode):
        pass

    def tok(self, text, max_seq_length):
        return tokenize(self.tokenizer, text, max_seq_length)
    
    def __call__(self, examples):
        if self.mode in ('filter', 'rerank'):
            return self.filter_rerank_call(examples)

        else:
            return self.default_call(examples)

    def default_call(self, examples):
        sentences = []
        quad_seqs = []
        chose_labels  = []
        binary_labels = []

        for example in examples:
            _sentences, candidates, perfered = self.get_sentence_candidates(example)

            chose_labels.append(perfered)
            for i, candidate in enumerate(candidates):
                sentences.append(_sentences[i])
                quad_seqs.append(candidate)
                binary_labels.append(2 if perfered == -100 else int(perfered == i))

        batch_encodings = self.tok(sentences, -1)
        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']        

        batch_encodings = self.tok(quad_seqs, -1)
        quad_labels = batch_encodings['input_ids']
        quad_labels = torch.tensor([
            [(l if l != self.tokenizer.pad_token_id else -100)
             for l in label]
            for label in quad_labels
        ])

        if self.max_seq_length > 0:
            input_ids = input_ids[:, :self.max_seq_length]
            attention_mask = attention_mask[:, :self.max_seq_length]
            if quad_labels is not None:
                quad_labels = quad_labels[:, :self.max_seq_length]

        chose_labels  = torch.tensor(chose_labels, dtype=torch.long)
        binary_labels = torch.tensor(binary_labels, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'quad_labels':  quad_labels,
            'chose_labels': chose_labels,
            'binary_labels': binary_labels,
            'examples': examples,
        }

    def get_sentence_candidates(self, example):
        candidates = [quads_seq.strip() for quads_seq in example['quad_preds']]
        sentences = example['sentences'] if 'sentences' in example else [example['sentence']] * 4

        if 'perfered_option' in example:
            perfered = example['perfered_option']
            if perfered == 4:
                quad_human = example['quad_human']
                if self.mode == 'train':
                    candidates = [quad_human] + random.sample(candidates, k=3)
                else:
                    candidates = [quad_human] + candidates[:3]
                perfered = 0

            if perfered == 5:
                perfered = -100

        else:
            perfered = None
        
        example['candidates'] = candidates
        return sentences, candidates, perfered

    def filter_rerank_call(self, examples):
        sentences = []
        quad_seqs = []

        for example in examples:
            _sentences, candidates, perfered = self.get_sentence_candidates(example)

            for sentence, candidate in zip(_sentences, candidates):
                sentences.append(sentence)
                quad_seqs.append(candidate)

        batch_encodings = self.tok(sentences, -1)
        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']        

        batch_encodings = self.tok(quad_seqs, -1)
        quad_labels = batch_encodings['input_ids']
        quad_labels = torch.tensor([
            [(l if l != self.tokenizer.pad_token_id else -100)
             for l in label]
            for label in quad_labels
        ])

        if self.max_seq_length > 0:
            input_ids = input_ids[:, :self.max_seq_length]
            attention_mask = attention_mask[:, :self.max_seq_length]
            if quad_labels is not None:
                quad_labels = quad_labels[:, :self.max_seq_length]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'quad_labels':  quad_labels,
            'examples': examples,
        }

        


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
        quad_subname: str='',
        subname: str='',
        model_name_or_path: str='',
        objective: str='list',
        alpha: float=1.,
        beta: float=1.,
        setting: str='01234+',
        k: int = 1000,
    ):

        super().__init__()

        self.model = RewardModel.from_pretrained(self.model_name_or_path)
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

    def forward(self, batch):
        scores = self.model.get_rewards(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_labels=batch['quad_labels']
        )
        binary_labels = batch['binary_labels']
        chose_labels  = batch['chose_labels']

        loss = 0
        accu = 0
        f1 = 0
        if self.training:
            loss = - self.alpha * scores[(binary_labels==1)+(binary_labels==2)].mean()

            if (chose_labels!=-100).sum() > 0:
                if self.objective == 'list':
                    loss += F.cross_entropy(scores.view(-1, 4), chose_labels)

                elif self.objective == 'pair1':
                    from utils.loss import dpo_loss_wo_ref
                    loss += dpo_loss_wo_ref(scores.view(-1, 4)[chose_labels!=-100], chose_labels[chose_labels!=-100], beta=self.beta)

                elif self.objective == 'pair2':
                    from utils.loss import rrhf_loss
                    loss += rrhf_loss(scores.view(-1, 4)[chose_labels!=-100], chose_labels[chose_labels!=-100])

                elif self.objective == 'point':
                    loss_point = -scores[(binary_labels==1)].mean()
                    loss_point = -(-scores[(binary_labels==0)].exp()).log1p().mean() * 3
                    loss += loss_point

                elif self.objective == 'none':
                    pass

        else:
            accu = self.cal_accuracy(scores, chose_labels)

        return {
            'scores': scores,
            'loss': loss,
            'accu': accu,
        }

    def cal_accuracy(self, scores, chose_labels):
        scores = scores.view(-1, 4)
        preds = scores.argmax(dim=-1)
        accu  = accuracy_score(chose_labels[chose_labels!=-100].cpu(), preds[chose_labels!=-100].cpu())
        return accu

    def training_step(self, batch, batch_idx):
        output = self(batch)

        loss = output['loss']
        accu = output['accu']

        self.log('accu', accu, prog_bar=True)

        return loss

    def eval_step(self, batch, batch_idx):
        output = self(batch)  
        scores = output['scores']

        chose_labels = batch['chose_labels'].cpu()
        binary_labels= batch['binary_labels'].cpu()

        return {
            'scores': scores,
            'chose_labels' : chose_labels,
            'binary_labels': binary_labels,
        }

    def validation_step(self, batch, batch_idx):
        output = self.eval_step(batch, batch_idx)
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self):
        self.current_val_metric = self.cal_metric(self.validation_step_outputs)

        if not hasattr(self, 'best_val_metric'):
            self.best_val_metric = self.current_val_metric
            self.save_model()

        elif self.best_val_metric <= self.current_val_metric:
            self.best_val_metric = self.current_val_metric
            self.save_model()

        self.validation_step_outputs.clear()

    def cal_metric(self, outputs):
        chose_labels = torch.cat([output['chose_labels'] for output in outputs])
        scores = torch.cat([output['scores'].view(-1) for output in outputs])

        accu = self.cal_accuracy(scores, chose_labels)
        return accu

    def on_train_end(self):
        accu = self.best_val_metric

        from datetime import datetime

        now = datetime.now()
        now = now.strftime("%Y-%m-%d")

        performance_file_name = os.path.join(self.output_dir, now, 'performance.txt')
        print('save performace to', performance_file_name)
        append_new_line(performance_file_name, json.dumps({
            'time': time.strftime('%Y-%m-%d %H_%M_%S', time.localtime()),
            'model_name_or_path': self.model_name_or_path,
            'subname': self.subname,
            'dataset': self.dataset,
            'seed': self.seed,
            'lr': self.learning_rate,
            'objective': self.objective,
            'k': self.k,
            'alpha': self.alpha,
            'beta': self.beta,
            'metric': accu,
            'setting': self.setting
        }))

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_test_epoch_end(self):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            scores = self.model.get_rewards(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                decoder_labels=batch['quad_labels'],
            )
            
        return {
            'scores': scores.view(-1, 4),
            'examples': batch['examples'],
        }

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
        print()
        print(f'CURRENT | accu: {pl_module.current_val_metric:.4f}', )
        print(f'BEST    | accu: {pl_module.best_val_metric:.4f}', )
        print()

    def on_test_end(self, trainer, pl_module):
        raise NotImplementedError

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):

        output_examples = []
        for output in tqdm(predictions):
            examples = output['examples']
            scores = output['scores']

            for example, four_score in zip(examples, scores):
                example['reward'] = four_score.tolist()
                output_examples.append(example)

        if 'quads' in output_examples[0] and output_examples[0]['quads'] is not None:
            detailed_metrics = self.cal_metric(output_examples)
            self.save_metric(detailed_metrics)

            from datetime import datetime

            now = datetime.now()
            now = now.strftime("%Y-%m-%d")

            dataset = self.name_space.model.dataset
            seed = self.name_space.model.seed
            subname = self.name_space.model.subname
            quad_subname = self.name_space.model.quad_subname

            output_dir = os.path.join(self.output_dir, now, f'{dataset}_{quad_subname}_{seed}_{subname}.json')
            print(f'save {len(output_examples)} to', output_dir)
            save_json(output_examples, output_dir)

        else:
            output_dir = os.path.join(self.output_dir, f'{self.name_space.model.dataset}.json')
            print(f'save {len(output_examples)} to', output_dir)
            save_json(output_examples, output_dir)

    def cal_metric(self, output_examples):
        n_precision = 0
        prec_hit    = 0
        n_recall    = 0
        recall_hit  = 0

        best_n_precision = 0
        best_prec_hit    = 0
        best_n_recall    = 0
        best_recall_hit  = 0

        rerank_n_precision = 0
        rerank_prec_hit    = 0
        rerank_n_recall    = 0
        rerank_recall_hit  = 0

        rerank_beam_indices = [0, 0, 0, 0, 0]
        best_beam_indices = [0, 0, 0, 0, 0]
        for example in output_examples:
            preds = example['quad_preds']
            true  = example['quads']

            _, np, ph, nr, rh = self._cal_metric(preds[0], true, example)
            n_precision += np
            prec_hit    += ph
            n_recall    += nr
            recall_hit  += rh

            ms = [self._cal_metric(pred, true, example)+(i, ) for i, pred in enumerate(preds)]

            r, rerank_perfered = max([(r,i) for i, r in enumerate(example['reward'])])    
            _, np, ph, nr, rh, _ = ms[rerank_perfered]
            rerank_beam_indices[rerank_perfered] += 1
            rerank_n_precision += np
            rerank_prec_hit    += ph
            rerank_n_recall    += nr
            rerank_recall_hit  += rh


            _, np, ph, nr, rh, _, i = max([self._cal_metric(pred, true, example)+(-i, i) for i, pred in enumerate(preds)])
            best_beam_indices[i]  += 1
            example['best'] = i
            best_n_precision += np
            best_prec_hit    += ph
            best_n_recall    += nr
            best_recall_hit  += rh

        precision, recall, f1 = cal_f1(prec_hit, n_precision, recall_hit, n_recall)
        rerank_precision, rerank_recall, rerank_f1 = cal_f1(rerank_prec_hit, rerank_n_precision, rerank_recall_hit, rerank_n_recall)
        best_precision, best_recall, best_f1 = cal_f1(best_prec_hit, best_n_precision, best_recall_hit, best_n_recall)
        
        detailed_metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'rerank_precision': rerank_precision,
            'rerank_recall': rerank_recall,
            'rerank_f1': rerank_f1,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'best_f1': best_f1
        }
        for metric_names in (('precision', 'recall', 'f1'), ('rerank_precision', 'rerank_recall', 'rerank_f1'), ('best_precision', 'best_recall', 'best_f1')):
            for metric_name in metric_names:
                value = detailed_metrics[metric_name] if metric_name in detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()

        print('rerank_beam_indices', rerank_beam_indices)
        print('best_beam_indices', best_beam_indices)

        return detailed_metrics    

    def _cal_metric(self, pred, true, example):
        split = lambda string: [s.strip() for s in string.split(';')]

        pred = parse_quads_seq(pred, example)[0]
        # true = parse_quads_seq(true, example)[0]

        n_precision = len(pred)
        prec_hit    = 0
        n_recall    = len(true)
        recall_hit  = 0

        for p in pred:
            if p in true:
                prec_hit += 1

        for t in true:
            if t in pred:
                recall_hit += 1

        precision, recall, f1 = cal_f1(prec_hit, n_precision, recall_hit, n_recall)

        return (f1, -n_precision), n_precision, prec_hit, n_recall, recall_hit

    def save_metric(self, detailed_metrics):
        from datetime import datetime

        now = datetime.now()
        now = now.strftime("%Y-%m-%d")

        performance_file_name = os.path.join(self.output_dir, now, 'performance.txt')
        print('save performace to', performance_file_name)
        append_new_line(performance_file_name, json.dumps({
            'time': time.strftime('%Y-%m-%d %H_%M_%S', time.localtime()),
            'model_name_or_path': None,
            'subname': self.name_space.model.quad_subname,
            'quad_subname': self.name_space.model.quad_subname,
            'scorer_subname': self.name_space.model.subname,
            'dataset': self.name_space.model.dataset,
            'seed': self.name_space.model.seed,
            'lr': self.name_space.model.learning_rate,
            'metric': {
                'before_precision': detailed_metrics['precision'],
                'before_recall': detailed_metrics['recall'],
                'before_f1': detailed_metrics['f1'],
                'precision': detailed_metrics['rerank_precision'],
                'recall': detailed_metrics['rerank_recall'],
                'f1': detailed_metrics['rerank_f1'],
                'best_precision': detailed_metrics['best_precision'],
                'best_recall': detailed_metrics['best_recall'],
                'best_f1': detailed_metrics['best_f1']
            }
        }))



def cal_f1(prec_hit, n_precision, recall_hit, n_recall):
    precision = prec_hit / n_precision if n_precision else 1
    recall    = recall_hit / n_recall if n_recall else 1
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return precision, recall, f1




def cli_main():
    from lightning.pytorch.cli import LightningCLI
    cli = LightningCLI(LightningModule, DataModule, CustomWriter)



if __name__ == '__main__':
    cli_main()
