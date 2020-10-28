import os
import io
import boto3
import json
import csv
import base64
import numpy as np

ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    data = json.loads(json.dumps(event))
    payload = data['body']

    encoded = base64.decodebytes(payload.encode('utf-8'))
    img = bytearray(encoded)

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/octet-stream',
        Body=img
    )

    result = json.loads(response['Body'].read().decode())
    breed_index = int(np.argmax(result))
    breed_name = get_breed(breed_index)

    json_response = {
        'isBase64Encoded': False,
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps({
            'breed_name': breed_name.strip('\n'),
            'breed_index': breed_index
        })
    }

    return json_response

def get_breed(index):
    breeds_list = []

    with open("breeds_dog.txt", "r") as dog_breeds:
        breeds_list = dog_breeds.readlines()

    return breeds_list[index]
