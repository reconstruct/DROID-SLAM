import sys
sys.path.append('.')
from argparse import ArgumentParser
from scripts.client import ReconstructClient

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--bucket', type=str, default='projmanager-development')
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    client = ReconstructClient(args.bucket)
    client.get_pointcloud_data(args.prefix, args.output_path)
