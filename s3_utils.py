"""
Amazon suggests, for objects larger than 100 MB,
customers should consider using the Multipart Upload capability.

This is a demo of Multipart Upload/Download using AWS Python SDK -boto3 library.
This module provides high level abstractions for efficient
uploads/download.
* Automatically switching to multipart transfer when
  a file is over a specific size threshold
* Uploading/downloading a file in parallel
* Progress callbacks to monitor transfers
Written By: ANKHI PAUL

https://github.com/ankhipaul/aws_demos
https://medium.com/analytics-vidhya/aws-s3-multipart-upload-download-using-boto3-python-sdk-2dedb0945f11

EDITS: corrections on the syntax of accessing files
       added some metadata on the file being uploaded
       use environment variables for AWS KEYS
TODO: option to loop through files in a directory for multiple uploads
      option to loop through folders in S3 to download all files
      prompt when file already exists in bucket, and will be overwritten
      error handling
      logging
Edited by: Maria Rosario Sebastian
Modifications
    - added input parameters (argparse)
    - uploads and downloads multiple files from a folder instead of one file, mentioned as TODO above.
    - user metadata
"""
import boto3
from boto3.s3.transfer import TransferConfig
import os
import threading
import sys
from datetime import datetime
import argparse
import glob
import requests

#get aws access keys from environment variables
s3_resource = boto3.resource(
    service_name='s3',
    region_name=os.environ["AWS_REGION"],
    aws_access_key_id=os.environ["AWS_KEY"],
    aws_secret_access_key=os.environ["AWS_SECRET_KEY"]
)

# User Metadata to include in uploading to S3 Bucket
user_name = os.environ['USERNAME']
ext_ip = requests.get('https://checkip.amazonaws.com').text.strip()

# multipart_threshold : Ensure that multipart uploads/downloads only happen if the size of a transfer
# is larger than 25 MB
# multipart_chunksize : Each part size is of 25 MB
config = TransferConfig(multipart_threshold=1024 * 25,
                        max_concurrency=10,
                        multipart_chunksize=1024 * 25,
                        use_threads=True)

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            #time.sleep(2)
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

'''Uploads all files from local file_path to S3 bucket_path (has to be in the format s3bucket/folder)'''
def multipart_upload_boto3(file_path, bucket_path):
    bucket_name, bucket_folder = os.path.split(bucket_path)
    files_to_upload = glob.glob(file_path + '/*')

    for file_to_upload in files_to_upload:
        now = datetime.now()
        #print("now =", now)
        # dd/mm/YY-H:M:S
        #dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")
        file_split=file_to_upload.replace("\\","/")
        dest_file_name=file_split.split('/')[-1]
        key=bucket_folder+'/'+dest_file_name
        file_to_upload_path=file_path+'/'+dest_file_name
        print('key: ', key)
        s3_resource.Object(bucket_name, key).upload_file(file_to_upload_path,
                                #added some metadata in the ExtraArgs
                                ExtraArgs={'Metadata': {'IP':ext_ip,'Sender': user_name, 'time': now.strftime("%d/%m/%Y-%H:%M:%S")}},
                                Config=config,
                                Callback=ProgressPercentage(file_to_upload_path)
                                )

'''This function downloads files from S3 bucket folder, format bucket/folder
    Files are downloaded in the out_path
'''
def multipart_download_boto3(bucket_path, out_path):
    bucket_name, bucket_folder = os.path.split(bucket_path)
    s3_bucket = s3_resource.Bucket(bucket_name)

    for object_summary in s3_bucket.objects.filter(Prefix=bucket_folder):
        print(object_summary.key)
        s3_key = object_summary.key
        path, filename = os.path.split(s3_key)
        
        if len(filename) > 0 :
            file_path_and_name = out_path + '/' + filename
            #you need to indicate the absolute path, which is needed to append the out_path folder
            file_path1 = os.path.dirname(os.path.abspath("__file__"))
            s3_resource.Object(bucket_name, s3_key).download_file(file_path_and_name,
                                Config=config,
                                Callback=ProgressPercentage(file_path1)
                                )

def main(args):
    UTIL_OPTION = args['opt']
    PATH_FROM = args['from']
    PATH_TO = args['to']

    if UTIL_OPTION.lower() == 'upload':
        print('upload')
        multipart_upload_boto3(PATH_FROM, PATH_TO)
    elif UTIL_OPTION.lower() == 'download':
        print('download')
        multipart_download_boto3(PATH_FROM, PATH_TO)
    else:
        print("Please specify either 'upload' or 'download' in --opt parameter.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', required=True, help='option, either upload or download')
    parser.add_argument('--from', required=True, help='path\\to\\files\\location')
    parser.add_argument('--to', required=True, help='path\\to\\files\\destination')
    args = vars(parser.parse_args())

    main(args)
