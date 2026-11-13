"""
Automated downloader for PerfectPetClub survey dataset from University of Hertfordshire S3 storage.

Security Policy:
- Credentials MUST NOT be hardcoded.
- Credentials are loaded securely from environment variables or a local .env file.
- .env is excluded from git tracking via .gitignore.
- Never prints or exposes secret keys or access tokens.
"""

import os
import sys
from pathlib import Path
import boto3
from botocore.client import Config
from botocore.exceptions import ConnectTimeoutError, EndpointConnectionError, ClientError

# Attempt to load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    # Look for .env in current directory or project root
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

ENDPOINT_URL = os.environ.get("UH_S3_ENDPOINT_URL", "http://vuh-la-specs-s3-1.herts.ac.uk:9000")
ACCESS_KEY = os.environ.get("UH_S3_ACCESS_KEY")
SECRET_KEY = os.environ.get("UH_S3_SECRET_KEY")
BUCKET_NAME = os.environ.get("UH_S3_BUCKET_NAME", "comsci-student-s3-1")
OBJECT_KEY = "PerfectPetClub_survey.csv"

raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
output_path = raw_dir / OBJECT_KEY


def download():
    """Download survey dataset from S3 with secure credential handling."""
    if not ACCESS_KEY or not SECRET_KEY:
        print("[ERROR] UH S3 credentials not found in environment!")
        print("Please configure UH_S3_ACCESS_KEY and UH_S3_SECRET_KEY in your .env file or environment.")
        print("See .env.example for required configuration format.")
        sys.exit(1)

    print(f"Connecting to UH S3 storage at {ENDPOINT_URL}...")
    print(f"Target Bucket: '{BUCKET_NAME}', Target Object: '{OBJECT_KEY}'")

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            config=Config(signature_version="s3v4", connect_timeout=5, read_timeout=15),
        )

        print(f"Querying bucket contents...")
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        contents = response.get("Contents", [])
        print(f"Found {len(contents)} object(s) in bucket:")
        for obj in contents:
            print(f"  • {obj['Key']} ({obj['Size']} bytes)")

        print(f"\nDownloading '{OBJECT_KEY}' to {output_path.name}...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET_NAME, OBJECT_KEY, str(output_path))
        print(f"Successfully downloaded '{OBJECT_KEY}' ({output_path.stat().st_size} bytes).")
    except (ConnectTimeoutError, EndpointConnectionError):
        print("\n[ERROR] Connection timed out!")
        print("Please ensure your University of Hertfordshire (UH) VPN is active and connected.")
        sys.exit(1)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        print(f"\n[ERROR] S3 Client Error [{error_code}]. Verify access credentials and permissions.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {type(e).__name__}")
        sys.exit(1)


if __name__ == "__main__":
    download()
