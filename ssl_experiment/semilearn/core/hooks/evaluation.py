import os
import json
from .hook import Hook

class EvaluationHook(Hook):
    
    def after_train_step(self, algorithm):
        warm_up_iter = getattr(algorithm, 'warm_up_iter', 0)
        if algorithm.it < warm_up_iter:
            every = max(1, getattr(algorithm, 'warm_up_eval_every', 1))
            if algorithm.it % every == every - 1:
                algorithm.print_fn(f"warm_up snapshot at it {algorithm.it + 1}...")
                algorithm.log_dict.update(algorithm.warm_up_evaluate('warm_up_eval'))
            return
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")
            eval_dict = algorithm.evaluate('eval')
            algorithm.log_dict.update(eval_dict)

            for k in algorithm.log_dict.keys():
                if 'top-1-acc' in k and algorithm.log_dict[k] > algorithm.best_eval_acc:
                    algorithm.best_eval_acc = algorithm.log_dict[k]
                    algorithm.best_it = algorithm.it
            try:
                rec = {'it': algorithm.it + 1}
                rec.update({k: float(v) for k, v in algorithm.log_dict.items()
                            if ('eval' in k or 'acc' in k) and isinstance(v, (int, float))})
                path = os.path.join(algorithm.save_dir, algorithm.save_name, 'eval_log.jsonl')
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'a') as fp:
                    fp.write(json.dumps(rec) + '\n')
            except Exception as e:
                algorithm.print_fn(f'eval_log write failed: {e}')

    
    def after_run(self, algorithm):
        
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            algorithm.save_model('latest_model.pth', save_path)

        results_dict = {'eval/best_acc': algorithm.best_eval_acc, 'eval/best_it': algorithm.best_it}
        if 'test' in algorithm.loader_dict:
            best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
            algorithm.load_model(best_model_path)
            test_dict = algorithm.evaluate('test')
            results_dict['test/best_acc'] = test_dict['test/top-1-acc']
        algorithm.results_dict = results_dict
        
