import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='fountain-P11/images')
    args = parser.parse_args()
    return args
