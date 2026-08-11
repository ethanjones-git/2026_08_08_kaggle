import os
import json
import time
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from requests.exceptions import HTTPError
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleS3Importer:

    def __init__(self, bucket_name="2026-08-08-kaggle-bucket", s3_prefix="kaggle_imports/rsna-knee-abnormality-detection"):
        self.s3_client = boto3.client('s3')
        self.S3_BUCKET_NAME = bucket_name
        self.S3_PREFIX = s3_prefix
        
        GB = 1024 * 1024 * 1024
        self.transfer_config = TransferConfig(
            multipart_threshold=1 * GB,
            multipart_chunksize=1 * GB,
            max_concurrency=4
        )

    def load_kaggle_credentials(self, secret_name="kaggleapi"):
        print(f"Fetching secret '{secret_name}' from Secrets Manager...")
        secrets_client = boto3.client('secretsmanager')
        response = secrets_client.get_secret_value(SecretId=secret_name)
        secret_dict = json.loads(response['SecretString'])

        os.environ['KAGGLE_USERNAME'] = secret_dict['username']
        kaggle_token = secret_dict.get('key') or secret_dict.get('api_key') or secret_dict.get('KAGGLE_KEY') or secret_dict.get('token')
        os.environ['KAGGLE_KEY'] = kaggle_token
        os.environ['KAGGLE_API_TOKEN'] = kaggle_token

    def import_file_to_s3(self, full_local_path, relative_path):
        s3_key = os.path.join(self.S3_PREFIX, relative_path).replace("\\", "/")
        try:
            self.s3_client.upload_file(
                Filename=full_local_path,
                Bucket=self.S3_BUCKET_NAME,
                Key=s3_key,
                Config=self.transfer_config
            )
            os.remove(full_local_path)
            print(f"Uploaded & deleted: {relative_path}")
        except ClientError as e:
            print(f"Failed to upload {s3_key}: {e}")

    def fetch_page_with_retry(self, api, competition_name, page_token=None, delay=1.0, max_retries=5):
        """Fetch a single page of files with built-in retry backoff on 429 errors."""
        kwargs = {"page_token": page_token} if page_token else {}
        
        for attempt in range(max_retries):
            try:
                # Politeness delay before calling API
                time.sleep(delay)
                return api.competition_list_files(competition_name, **kwargs)
            except HTTPError as err:
                if err.response is not None and err.response.status_code == 429:
                    wait_time = (2 ** attempt) * 3  # Exponential backoff: 3s, 6s, 12s, 24s...
                    print(f"\nRate limited (429). Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise err
        
        raise Exception(f"Exceeded max retries fetching page for token: {page_token}")

    def run(self, competition_name="rsna-knee-abnormality-detection", download_dir="./kaggle_temp", page_num = 1):

        self.load_kaggle_credentials()
        
        api = KaggleApi()
        api.authenticate()

        print(f"Starting page-by-page import for '{competition_name}'...")
        os.makedirs(download_dir, exist_ok=True)
        
        page_token = None
        total_uploaded = 0

        while True:
            print(f"\n--- Fetching metadata for Page {page_num} ---")
            
            # Fetch current page with backoff protection
            response = self.fetch_page_with_retry(api, competition_name, page_token=page_token, delay=1.0)

            # Extract list of files on this page
            if hasattr(response, "files"):
                page_files = response.files
            elif isinstance(response, dict) and "files" in response:
                page_files = response["files"]                
            else:
                page_files = response

            if not page_files:
                print("No files found on this page.")
                break

            print(f"Page {page_num}: Found {len(page_files)} files to process.")
            
            # Process every file on the current page immediately
            for idx, file_item in enumerate(page_files, 1):
                if hasattr(file_item, "name"):
                    file_name = file_item.name
                elif isinstance(file_item, dict):
                    file_name = file_item.get("name") or file_item.get("ref")
                else:
                    file_name = str(file_item)

                if not file_name:
                    continue

                local_target_path = os.path.join(download_dir, file_name)
                os.makedirs(os.path.dirname(local_target_path), exist_ok=True)

                if 'train' in file_name:
                    print(f"[Page {page_num} | File {idx}/{len(page_files)}] Downloading '{file_name}'...")
                    try:
                        api.competition_download_file(
                            competition=competition_name,
                            file_name=file_name,
                            path=os.path.dirname(local_target_path)
                        )
                        time.sleep(3)

                        if os.path.exists(local_target_path):
                            self.import_file_to_s3(local_target_path, file_name)
                            total_uploaded += 1
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
                
            # Extract next page token
            next_token = None
            if hasattr(response, "next_page_token"):
                next_token = response.next_page_token
            elif isinstance(response, dict):
                next_token = response.get("nextPageToken") or response.get("next_page_token")

            if not next_token:
                print("\nReached final page!")
                break

            page_token = next_token
            page_num += 1
            time.sleep(3)


        print(f"\nFinished! Processed {total_uploaded} files across {page_num} pages.")