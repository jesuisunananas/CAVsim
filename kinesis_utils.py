import os
import boto3
from dotenv import load_dotenv

load_dotenv()

def get_kvs_hls_url(stream_name, region_name="us-west-2"):
    """
    Fetches a live HLS streaming session URL for a given Kinesis Video Stream.
    """
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    kvs_client = boto3.client(
        'kinesisvideo',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    endpoint_response = kvs_client.get_data_endpoint(
        StreamName=stream_name,
        APIName='GET_HLS_STREAMING_SESSION_URL'
    )
    endpoint_url = endpoint_response['DataEndpoint']

    kvs_media_client = boto3.client(
        'kinesis-video-archived-media',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
        endpoint_url=endpoint_url
    )

    url_response = kvs_media_client.get_hls_streaming_session_url(
        StreamName=stream_name,
        PlaybackMode='LIVE'
    )

    return url_response['HLSStreamingSessionURL']
