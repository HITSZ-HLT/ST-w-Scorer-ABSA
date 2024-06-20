from utils import load_line_json, yield_data_file
from collections import defaultdict
import os



def main(args):
    args.seed = None

    print(args.seed_num)

    if args.seed_num == 10:
        seeds = (42, 52, 62, 72, 82, 142, 152, 162, 172, 182)
    elif args.seed_num == 5:
        seeds = (42, 52, 62, 72, 82)

    if type(args.source_file) is str:
        metrics = load_line_json(args.source_file)
    
    elif type(args.source_file) is list:
        metrics = []
        for source_file in args.source_file:
            metrics.extend(list(load_line_json(source_file)))

    datasets = args.datasets.split()

    metric_dict = defaultdict(dict)
    subnames = set()
    for metric in metrics:
        if (metric['subname'] in args.subname or len(args.subname)==0) and (metric['seed'] == args.seed or args.seed is None):
            dataset = metric['dataset']
            
            if dataset not in datasets:
                continue

            seed    = metric['seed']
            metric_dict[metric['subname']][(dataset, seed)] = (metric['metric'], metric['model_name_or_path'])
            subnames.add(metric['subname'])

    # subnames = sorted(subnames, key=lambda subname: float(subname[1:]))
    subnames = sorted(subnames)
    print(subnames)

    print('\t' + ' '.join(datasets))
    for subname in subnames:
        if not args.detail:
            print('-----------------------------------------------')
        else:
            print('------------------------------------------------------------------------------------')
        print(subname, end='')
        stat = {}
        for seed in seeds:
            # oversize
            print(f'\t{seed}', end='\t')
            for dataset in datasets:
                if (dataset, seed) in metric_dict[subname]:
                    metric, model = metric_dict[subname][(dataset, seed)]
                    if not args.detail:
                        print(f"{round(metric['f1']*10000)}", end=' ')
                    else:
                        print(f"{round(metric['precision']*10000)} {round(metric['recall']*10000)} {round(metric['f1']*10000)}", end='  ')
                
                else:
                    if not args.detail:
                        print("0000", end=' ')
                    else:
                        print("0000 0000 0000", end='  ')
            print()

        print(f'\taverage', end='\t')
        sum_ = 0
        for dataset in datasets:
            ms = [metric_dict[subname][(dataset, seed)][0]['f1'] for seed in seeds if (dataset, seed) in metric_dict[subname]]
            if args.detail:
                ps = [metric_dict[subname][(dataset, seed)][0]['precision'] for seed in seeds if (dataset, seed) in metric_dict[subname]]
                rs = [metric_dict[subname][(dataset, seed)][0]['recall'] for seed in seeds if (dataset, seed) in metric_dict[subname]]
            av = sum(ms) / len(ms) if len(ms) > 0 else 0
            sum_ += av
            
            if not args.detail:
                print(f"{round(av*10000):4d}", end=' ')
            else:
                p_av = sum(ps) / len(ps) if len(ps) > 0 else 0
                r_av = sum(rs) / len(rs) if len(rs) > 0 else 0
                print(f"{round(p_av*10000):4d} {round(r_av*10000):4d} {round(av*10000):4d}", end='  ')

        print(f"{round(sum_/4*10000)}")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='../output/quad/')
    parser.add_argument('--date', type=str, default='')
    parser.add_argument('--subname', type=str, default='')
    parser.add_argument('--datasets', type=str, default='acos/laptop16 acos/rest16 asqp/rest15 asqp/rest16')
    parser.add_argument('--seed_num', type=int, default=5)
    parser.add_argument('--detail', action='store_true')
    

    args = parser.parse_args()
    args.subname = args.subname.split()

    if args.date == '':
        args.source_file = []
        for dir_ in yield_data_file(args.output_dir):
            file_name = os.path.join(dir_, 'performance.txt')
            if os.path.exists(file_name):
                args.source_file.append(file_name)
                print(file_name)

    else:
        args.source_file = os.path.join(args.output_dir, args.date, 'performance.txt')
        print(args.source_file)

    main(args)
