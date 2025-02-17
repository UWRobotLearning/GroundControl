import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config
import os
import tqdm
from groundcontrol_assets import GROUNDCONTROL_ASSETS_DATA_DIR
from dataclasses import dataclass, MISSING

@dataclass
class S3BucketCfg:
    bucket_name: str = MISSING
    region_name: str = MISSING
    s3_folder_path: str = MISSING
    target_local_path: str = MISSING
    my_access_key_ID: str = None
    my_secret_key: str = None

@dataclass
class RobotAssetBucketCfg(S3BucketCfg):
    bucket_name: str = "omniverse-content-production"
    region_name: str = "us-west-2"
    s3_folder_path: str = "Assets/Isaac/4.5/Isaac/Robots/BostonDynamics/spot/spot.usd"
    target_local_path: str = f"{GROUNDCONTROL_ASSETS_DATA_DIR}/Robots/"

@dataclass
class WorldAssetBucketCfg(S3BucketCfg):
    bucket_name: str = "arl-tiamat-data"
    region_name: str = "us-east-1"
    s3_folder_path: str = "Collected_GQ_lite/"
    target_local_path: str = f"{GROUNDCONTROL_ASSETS_DATA_DIR}/Worlds/"
    my_access_key_ID: str = None 
    my_secret_key: str = None 

    def __post_init__(self):
        # check if my_access_key is missing
        if not self.my_access_key_ID:
            print("Missing Access Key ID in config dataclass. Add manually or enter here: \n")
            self.my_access_key_ID = input("Enter your AWS Access Key ID: ")
        if not self.my_secret_key:
            print("Missing Secret Access Key in config dataclass. Add manually or enter here: \n")
            self.my_secret_key = input("Enter your AWS Secret Access Key: ")

def download_s3_folder(bucket_name, region_name, s3_folder, local_dir, aws_access_key_id=None, aws_secret_access_key=None):
    """
    Downloads an entire folder (prefix) from an S3 bucket.

    Args:
        bucket_name: The name of the S3 bucket.
        region_name (optional): The AWS region.
        s3_folder: The folder (prefix) within the S3 bucket to download.  e.g., 'my_folder/'
                    If you want to download the entire bucket, use an empty string ('').
        local_dir: The local directory where the folder's contents should be saved.
        aws_access_key_id: Your AWS access key ID.
        aws_secret_access_key: Your AWS secret access key.

    Returns:
        True if the download was successful (or if there was nothing to download), False otherwise.

    Raises:
        ValueError: If bucket_name is empty, or local_dir doesn't exist.
        botocore.exceptions.ClientError: If there are any issues with S3 access.
        botocore.exceptions.NoCredentialsError: If AWS credentials cannot be found.
        Exception: For other unforeseen errors.

    """

    if not bucket_name:
        raise ValueError("Bucket name cannot be empty.")

    if not os.path.isdir(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    try:
        # Create an S3 client (same credential handling as before)
        if aws_access_key_id and aws_secret_access_key:
            s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key,
                              region_name=region_name)
        else:
            s3 = boto3.client('s3', region_name=region_name)

        # List objects within the specified folder (prefix)
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)

        downloaded_something = False

        # First iterate over pages and objects to get the total size of all objects
        total_size_to_download = 0
        for page in pages:
            if 'Contents' in page:
                total_page_size = sum(obj['Size'] for obj in page['Contents'])
                total_size_to_download += total_page_size
        print(total_size_to_download)

        with tqdm.tqdm(total=total_size_to_download, unit="B", unit_scale=True, desc="Graces Quarters Assets") as pbar:
            for page in pages:
                if 'Contents' in page:  # Check if the page has any objects
                    for obj in page['Contents']:
                        s3_key = obj['Key']
                        # Construct the local file path, preserving the folder structure
                        relative_path = os.path.relpath(s3_key, s3_folder)
                        local_file_path = os.path.join(local_dir, relative_path)

                        # Create necessary directories
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                        # Download the file
                        try:
                            s3.download_file(
                                bucket_name,
                                s3_key,
                                local_file_path,
                                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
                            )
                            #print(f"Downloaded: {s3_key} -> {local_file_path}")
                            downloaded_something = True
                        except botocore.exceptions.ClientError as e:
                            print(f"Error downloading {s3_key}: {e}")
                            return False  # Return False on any download error

        if not downloaded_something:
            print(f"No files found in S3 folder '{s3_folder}' within bucket '{bucket_name}'.")
            return True # return true, since technically the download was successful - no downloads were needed

        return True  # Return True if everything was downloaded successfully

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "NoSuchBucket":
            print(f"Error: The bucket '{bucket_name}' does not exist.")
        else:
            print(f"Error accessing S3: {e}")
        return False
    except botocore.exceptions.NoCredentialsError:
        print("Error: No AWS credentials found.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def download_s3_file(bucket_name, region_name, s3_file_key, local_dir, aws_access_key_id=None, aws_secret_access_key=None):

    try:
        # Create an S3 client.  Prioritize explicit credentials, then profile, then environment/IAM.
        if aws_access_key_id and aws_secret_access_key:
            s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key,
                              region_name=region_name)
        else:
             # Use UNSIGNED credentials (environment variables, IAM role, etc.)
            s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

        # split filepath to include everything after "Robots/"
        filepath_after_robots = s3_file_key.split("Robots/")[1]

        # append file name to local_dir
        local_target_filepath = os.path.join(local_dir, filepath_after_robots)

        # Create necessary directories
        os.makedirs(os.path.dirname(local_target_filepath), exist_ok=True)

        # Download the file
        s3.download_file(bucket_name, s3_file_key, local_target_filepath)
        print(f"File '{s3_file_key}' downloaded from bucket '{bucket_name}' to '{local_target_filepath}'")
        return True


    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"Error: The object '{s3_file_key}' does not exist in bucket '{bucket_name}'.")
        elif e.response['Error']['Code'] == "403":
            print(f"Error: Access denied to '{s3_file_key}' in bucket '{bucket_name}'. Check your permissions.")
        else:
            print(f"Error downloading file from S3: {e}")
        return False
    except botocore.exceptions.NoCredentialsError:
        print("Error: No AWS credentials found.  Please configure your credentials.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


if __name__ == '__main__':

    robot_cfg = RobotAssetBucketCfg()
    if download_s3_file(
        robot_cfg.bucket_name,
        robot_cfg.region_name,
        robot_cfg.s3_folder_path,
        robot_cfg.target_local_path
    ):
        print("Robot asset download successful.")

    world_cfg = WorldAssetBucketCfg()
    if download_s3_folder(
        world_cfg.bucket_name,
        world_cfg.region_name,
        world_cfg.s3_folder_path,
        world_cfg.target_local_path,
        aws_access_key_id=world_cfg.my_access_key_ID,
        aws_secret_access_key=world_cfg.my_secret_key
    ):
        print("World asset download successful.")