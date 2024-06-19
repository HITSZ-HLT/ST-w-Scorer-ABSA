from utils.quad import parse_quads_seq
from utils import append_new_line, save_json
import os, time, json



class F1_Measure:
    def __init__(self):
        self.pred_list = []
        self.true_list = []

    def pred_inc(self, idx, preds):
        for pred in preds:
            self.pred_list.append((idx, pred))
            
    def true_inc(self, idx, trues):
        for true in trues:
            self.true_list.append((idx, true))
            
    def report(self):
        self.f1, self.p, self.r = self.cal_f1(self.pred_list, self.true_list)
        return self.f1
    
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise NotImplementedError

    def cal_f1(self, pred_list, true_list):
        n_tp = 0
        for pred in pred_list:
            if pred in true_list:
                n_tp += 1    
        _p = n_tp / len(pred_list) if pred_list else 1
    
        n_tp = 0
        for true in true_list:
            if true in pred_list:
                n_tp += 1 
        _r = n_tp / len(true_list) if true_list else 1

        f1 = 2 * _p * _r / (_p + _r) if _p + _r else 0

        return f1, _p, _r



class Result:
    def __init__(self, data):
        self.data = data 

    def __ge__(self, other):
        return self.monitor >= other.monitor

    def __gt__(self, other):
        return self.monitor >  other.monitor

    @classmethod
    def parse_from(cls, outputs):
        data = {}

        for output in outputs:
            examples = output['examples']
            predictions = output['predictions']

            for example, prediction in zip(examples, predictions):
                ID = example.pop('ID')
                quads = example.pop('quads') if 'quads' in example else parse_quads_seq(example.pop('quads_seq'))[0]

                data[ID] = {
                    'ID': ID,
                    'sentence': example['sentence'],
                    'quads': quads,
                    'quad_preds' : parse_quads_seq(prediction, example)[0],
                    **example
                }

        return cls(data)

    def cal_metric(self):
        f1 = F1_Measure()

        for ID in self.data:
            example = self.data[ID]
            g = example['quads']
            p = example['quad_preds']
            f1.true_inc(ID, g)
            f1.pred_inc(ID, p)

        f1.report()

        self.detailed_metrics = {
            'f1': f1['f1'],
            'recall': f1['r'],
            'precision': f1['p'],
        }

        self.monitor = self.detailed_metrics['f1']

    def save_prediction(self, output_dir, model_name_or_path, subname, dataset, seed, lr):
        from datetime import datetime

        now = datetime.now()
        now = now.strftime("%Y-%m-%d")
        file_name = os.path.join(output_dir, now, f'{dataset}_{subname}_{seed}.json')

        print('save prediction to', file_name)
        save_json(
            {
                'data': self.data,
                'meta': (model_name_or_path, subname, dataset, seed, lr, now)
            }, 
            file_name
        )

    def save_metric(self, output_dir, model_name_or_path, subname, dataset, seed, lr):

        from datetime import datetime

        now = datetime.now()
        now = now.strftime("%Y-%m-%d")

        performance_file_name = os.path.join(output_dir, now, 'performance.txt')
        print('save performace to', performance_file_name)
        append_new_line(performance_file_name, json.dumps({
            'time': time.strftime('%Y-%m-%d %H_%M_%S', time.localtime()),
            'model_name_or_path': model_name_or_path,
            'subname': subname,
            'dataset': dataset,
            'seed': seed,
            'lr': lr,
            'metric': self.detailed_metrics
        }))

    def report(self):
        for metric_names in (('precision', 'recall', 'f1'),):
            for metric_name in metric_names:
                value = self.detailed_metrics[metric_name] if metric_name in self.detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()
