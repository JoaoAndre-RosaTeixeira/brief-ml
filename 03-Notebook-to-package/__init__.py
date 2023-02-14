from data import Data
from trainer import Trainer


def run():
    data = Data("data/train.csv", 10000).clean_data()
    trainer = Trainer(data)
    trainer.run()
    print(trainer.evaluate())