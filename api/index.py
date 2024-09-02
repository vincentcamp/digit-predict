import json
import numpy as np
import sys
import os
import traceback

# Global variable declarations
W1 = None
b1 = None
W2 = None
b2 = None

def load_model_parameters():
    global W1, b1, W2, b2
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, 'digit_recognizer_model.json')

    try:
        with open(model_path, 'r') as f:
            model_params = json.load(f)
        print("Model parameters loaded successfully", file=sys.stderr)
        W1 = np.array(model_params['W1'])
        b1 = np.array(model_params['b1'])
        W2 = np.array(model_params['W2'])
        b2 = np.array(model_params['b2'])
    except Exception as e:
        print(f"Error loading model parameters: {str(e)}", file=sys.stderr)
        raise

# Load model parameters at module initialization
load_model_parameters()

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

def handler(event, context):
    try:
        if event['httpMethod'] == 'POST' and event['path'] == '/api/predict':
            body = json.loads(event['body'])
            image_data = np.array(body['image']).reshape(784, 1) / 255.0

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, image_data)
            prediction = int(get_predictions(A2)[0])

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'prediction': prediction})
            }
        else:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'Not Found'})
            }
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        print(f"Error traceback: {traceback.format_exc()}", file=sys.stderr)
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e), 'traceback': traceback.format_exc()})
        }

# This is only used for local testing
if __name__ == "__main__":
    # Simulate a Vercel serverless environment event
    test_event = {
        'httpMethod': 'POST',
        'path': '/api/predict',
        'body': json.dumps({
            'image': [0] * 784  # Replace with actual test data
        })
    }
    print(handler(test_event, None))