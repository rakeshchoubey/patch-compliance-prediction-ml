import json
import os
import boto3

client = boto3.client("sagemaker")


def lambda_handler(event, context):

    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    # Remove filename
    folder = os.path.dirname(key)
    # Replace raw with processed
    processed_folder = folder.replace("/raw/", "/processed/")
    processed_path = f"s3://{bucket}/{processed_folder}/"

    # Remove filename
    folder = key.rsplit("/", 1)[0] + "/"
    input_path = f"s3://{bucket}/{folder}"
    model_output_path = f"s3://{bucket}/{folder}/model/"
    response = client.start_pipeline_execution(
        PipelineName="PredictEC2PatchCompliancePipeline",
        PipelineParameters=[
            {
                "Name": "InputData",
                "Value": input_path
            },
            {
                "Name": "ProcessedOutput",
                "Value": processed_path
            },
            {
                "Name": "ModelOutput",
                "Value": model_output_path
            }
        ]
    )

    return response