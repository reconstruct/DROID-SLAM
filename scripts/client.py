from datetime import datetime, timedelta, date
import os
import boto3
import json
from pathlib import Path
from tqdm import tqdm

class ReconstructClient:
    def __init__(self, bucket='projmanager-beta'):
        self.s3 = boto3.client('s3')
        self.bucket = bucket

    def get_pointcloud_data_by_id(self, db, pcid, output_path):
        query = {'_id': ObjectId(pcid)}
        pc_data = self.db['pointclouds'].find(query)
        if len(pc_data) == 0:
            raise RuntimeError("Point Cloud Not Found!")
        pointcloud = pc_data[0]

        prefix = pointcloud['path']
        # remove edging slashes
        if prefix[0] == '/':
            prefix = prefix[1:]
        if prefix[-1] == '/':
            prefix = prefix[:1]

        return self.get_pointcloud_data(prefix, output_path)

    def get_pointcloud_data(self, prefix, output_path):
        ''' gets point cloud data given s3 prefix '''

        # create output directory
        output_path = Path(output_path)
        image_path = output_path / 'images'
        image_path.mkdir(parents=True, exist_ok=True)


        # 1. check if point cloud has video
        # video exists if there is s3://PID/PointCloud/PCID/video_metadata.json
        # video_metadata.json format

        data = self.s3.get_object(Bucket=self.bucket, Key=prefix + '/video_metadata.json')
        contents = data['Body'].read()
        video_metadata = json.loads( contents.decode("utf-8"))

        image_files = self.parse_video_metadata(video_metadata)

        sorted_image_files = sorted(image_files, key=lambda x: x[0])

        for i, (frame_time, image_file) in tqdm(enumerate(sorted_image_files), total=len(sorted_image_files), desc='downloading images'):
            out_image_filename = f'{i:08d}.jpg'
            self.s3.download_file(self.bucket, image_file, str(output_path / 'images' / out_image_filename))


        # 2. get calibration file
        data = self.s3.get_object(Bucket=self.bucket, Key=prefix + '/camera_calib.json')
        contents = data['Body'].read()
        camera_calib = json.loads( contents.decode("utf-8"))
        parsed_calib = self.parse_camera_calib(camera_calib)

        with open(output_path / 'calibration.txt', 'w') as f:
            f.write(' '.join([str(f) for f in parsed_calib]))


    def parse_video_metadata(self, metadata):
        ''' Given json format of metadata file, returns list of images '''
        image_files = []
        # time_format = '%Y:%m:%d %H:%M:%S'
        for video_name, video_metadata in metadata.items():
            for img_uuid, img_obj in video_metadata.items():
                # frame_time = datetime.strptime(img_obj['timeCreated'][:4, time_format)
                frame_time = int(img_obj['name'])
                image_files.append((frame_time, img_obj['path']))

        return image_files

    def parse_camera_calib(self, metadata):
        for calib in metadata.values():
            for k, v in calib.items():
                if k == 'project_id':
                    pcid = v
            pc_calib = calib[pcid]
            intrinsic = pc_calib['project_camera']['intrinsics']
        I = intrinsic
        W = I['width']
        H = I['width']
        f = I['focal']

        fx = f * max(W, H)
        fy = fx
        cx = W / 2.0
        cy = H / 2.0
        if ('k1' in I) and ('k2' in I) and ('p1' in I) and ('p2' in I):
            k1 = I['k1']
            k2 = I['k2']
            p1 = I['p1']
            p2 = I['p2']
            if 'k3' in I:
                k3 = I['k3']
                return [fx, fy, cx, cy, k1, k2, p1, p2, k3]
            else:
                return [fx, fy, cx, cy, k1, k2, p1, p2]
        else:
            return [fx, fy, cx, cy]
        
        return [fx, fy, cx, cy, k1, k2, p1, p2, k3]


                
