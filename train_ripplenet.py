import argparse
from recbole.quick_start import run_recbole

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='ripplenet_config.yaml')
    parser.add_argument('--dataset', type=str, default='music_dataset')
    args = parser.parse_args()
    
    # Additional config parameters
    config_dict = {
        'data_path': './dataset/',
        'checkpoint_dir': './checkpoint/',
        'log_wandb': False,  # Set to True if you want to use Weights & Biases
    }
    
    # Run RecBole
    run_recbole(
        model='RippleNet',
        dataset=args.dataset,
        config_file_list=[args.config],
        config_dict=config_dict
    )

if __name__ == '__main__':
    main()