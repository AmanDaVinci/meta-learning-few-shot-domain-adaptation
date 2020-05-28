import importlib
from meta_infomax.trainers.multitask_trainer import MultitaskTrainer
from meta_infomax.trainers.maml_trainer import MAMLTrainer
from meta_infomax.trainers.fomaml_trainer import FOMAMLTrainer
from meta_infomax.trainers.protonet_trainer import ProtonetTrainer


import argparse
import numpy as np

from meta_infomax.trainers.evaluation_trainer import EvaluationTrainer
from meta_infomax.trainers.maml_trainer import MAMLTrainer
from meta_infomax.trainers.multitask_trainer import MultitaskTrainer

MULTITASK_TRAINER = 'multitask'
MAML_TRAINER = 'maml'
PROTOTYPICAL_TRAINER = 'prototypical'
EVALUATION_TRAINER = 'evaluation'


def main():
    """ Runs the trainer based on the given experiment configuration """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs.evaluation_config", help='experiment configuration dict')
    parser.add_argument('--trainer', default=None, type=str, help='Type of trainer model')
    parser.add_argument('--exp_name', default=None, type=str, help='Experiment name. Results saved in this folder.')
    parser.add_argument('--train', default=False, action='store_true', help='whether to train')
    parser.add_argument('--test', default=False, action='store_true', help='whether to test')
    # TODO: implement no_cuda
    parser.add_argument("--no_cuda", action="store_true", help='Whether not to use CUDA when available')
    parser.add_argument("--lr", default=None, type=float, help='Learning rate. Overwrites config.')
    parser.add_argument("--k_shot", default=None, type=int, help='K in K-shot evaluation.')
    parser.add_argument("--batch_size", default=None, type=int, help='Train batch size.')
    parser.add_argument("--epochs", default=None, type=int, help='How many epochs to train for.')
    parser.add_argument("--validation_size", default=None, type=float, help='Percentage of train split to use as validation.')
    parser.add_argument("--num_examples", default=None, type=int,
                        help='How many examples the model should be trained on. Overrides epochs.')
    parser.add_argument("--test_same_domains", default=False, action='store_true',
                        help=('If present, validation is on test set of train domains. Can be used '
                              'to test overfitting on domains compared to data.'))
    parser.add_argument("--n_evaluations", default=None, type=int, help='How many times to evaluate on test domains.')
    # parser.add_argument('--test', type=bool, help='whether to test')
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    random_seeds = [40, 41, 42]
    for random_seed in random_seeds:
        config_module.config['seed'] = random_seed
        for arg_name, value in args.__dict__.items():
            # if a parameter is specified, overwrite config
            if value is not None:
                config_module.config[arg_name] = value
        if args.no_cuda:
            config_module.config['device'] = 'cpu'
        trainer_type = config_module.config['trainer']
        assert trainer_type in (MULTITASK_TRAINER, MAML_TRAINER, PROTOTYPICAL_TRAINER, EVALUATION_TRAINER), \
            'Make sure you have specified a correct trainer.'
        if trainer_type == MULTITASK_TRAINER:
            trainer = MultitaskTrainer(config_module.config)
        elif trainer_type == MAML_TRAINER:
            trainer = FOMAMLTrainer(config_module.config)
        elif trainer_type == PROTOTYPICAL_TRAINER:
            trainer = ProtonetTrainer(config_module.config)
        elif trainer_type == EVALUATION_TRAINER:
            trainer = EvaluationTrainer(config_module.config)

        if args.train:
            trainer.run()

        if args.test:
            ### only 1 version for now
            for unfrozen_layers in ["(10, 11)"]:
                for num_examples in ["3500", 'all']:
                    config_results = []
                    for random_seed in random_seeds:
                        ### loading best model from checkpoint
                        trainer.load_checkpoint(experiment_name = config_module.config['exp_name'], file_name = "unfrozen_bert:" + unfrozen_layers + "_num_examples:" + num_examples + "_seed:" + str(random_seed) + "_best-model.pt")
                        config_results.append(trainer.test())
                    #### averaging the seeds:
                    print("\n\nLayers " + str(unfrozen_layers) + " Data " + str(num_examples) + " Average results across seeds")
                    print("Mean: " + str(np.mean(config_results)))
                    print("SD: " + str(np.std(config_results)) + "\n\n")



if __name__ == "__main__":
    main()
