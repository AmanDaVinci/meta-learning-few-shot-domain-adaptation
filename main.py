import importlib
from meta_infomax.trainers.multitask_trainer import MultitaskTrainer
from meta_infomax.trainers.maml_trainer import MAMLTrainer
from meta_infomax.trainers.fomaml_trainer import FOMAMLTrainer


import argparse

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
    parser.add_argument('--trainer', default="MULTITASK_TRAINER", help='type of trainer model')

    parser.add_argument('--train', default=True, action='store_true', help='whether to train')
    # TODO: implement no_cuda
    parser.add_argument("--no_cuda", action="store_true", help='Whether not to use CUDA when available')
    # parser.add_argument('--test', type=bool, help='whether to test')
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    trainer_type = config_module.config['trainer']
    assert trainer_type in (MULTITASK_TRAINER, MAML_TRAINER, PROTOTYPICAL_TRAINER, EVALUATION_TRAINER), \
        'Make sure you have specified a correct trainer.'
    if trainer_type == MULTITASK_TRAINER:
        trainer = MultitaskTrainer(config_module.config)
    elif trainer_type == MAML_TRAINER:
        trainer = FOMAMLTrainer(config_module.config)
    elif trainer_type == PROTOTYPICAL_TRAINER:
        pass  # ProtoTrainer(config_module.config)
    elif trainer_type == EVALUATION_TRAINER:
        trainer = EvaluationTrainer(config_module.config)
    if args.train:
        trainer.run()
    # if args.test:
    #    test_report = trainer.test()
    #    print(test_report)


if __name__ == "__main__":
    main()
