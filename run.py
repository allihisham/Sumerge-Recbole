from recbole.config import Config
from recbole.data import create_dataset, data_preparation

if __name__ == '__main__':
    config = Config(model='RippleNet', dataset='music', config_file_list=['test.yaml'])
    dataset = create_dataset(config)