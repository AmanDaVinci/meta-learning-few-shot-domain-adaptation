import argparse
import importlib
from meta_infomax.trainers.multitask_trainer import MultitaskTrainer
from meta_infomax.trainers.maml_trainer import MAMLTrainer


def main():
    """ Runs the trainer based on the given experiment configuration """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs.multitask_config", help='experiment configuration dict')
    parser.add_argument('--train', default=True, help='whether to train')
    # parser.add_argument('--test', type=bool, help='whether to test')
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    trainer = MAMLTrainer(config_module.config)
    if args.train:
        trainer.run()
    # if args.test:
    #    test_report = trainer.test()
    #    print(test_report)

if __name__ == "__main__":
    main()