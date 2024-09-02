from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import sys
import os
import resource
import traceback

global W1, b1, W2, b2
W1 = None
b1 = None
W2 = None
b2 = None

def load_model_parameters():
    global W1, b1, W2, b2
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, 'digit_recognizer_model.json')

    print(f"Current directory: {current_dir}", file=sys.stderr)
    print(f"Model path: {model_path}", file=sys.stderr)

    try:
        print("Attempting to load model parameters...", file=sys.stderr)
        with open(model_path, 'r') as f:
            model_params = json.load(f)
        print("Model parameters loaded successfully", file=sys.stderr)

        print("Assigning model parameters...", file=sys.stderr)
        W1 = np.array(model_params['W1'])
        b1 = np.array(model_params['b1'])
        W2 = np.array(model_params['W2'])
        b2 = np.array(model_params['b2'])
        print("Model parameters assigned successfully", file=sys.stderr)

    except FileNotFoundError as e:
        print(f"Error: Model file not found - {str(e)}", file=sys.stderr)
        raise

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in model file - {str(e)}", file=sys.stderr)
        raise

    except KeyError as e:
        print(f"Error: Missing key in model parameters - {str(e)}", file=sys.stderr)
        raise

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

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((10, 1))
    one_hot_Y[Y] = 1
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def log_memory_usage():
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Convert to MB
    print(f"Current memory usage: {memory_usage:.2f} MB", file=sys.stderr)
    
    memory_limit = 1024  # 1024 MB = 1 GB
    if memory_usage > memory_limit * 0.9:  # Warning at 90% usage
        print(f"WARNING: Approaching memory limit. Usage: {memory_usage:.2f} MB / {memory_limit} MB", file=sys.stderr)

def handler(event, context):
    print("Received event:", event, file=sys.stderr)
    try:
        if event.get('httpMethod') == 'POST' and event.get('path') == '/api/predict':
            body = json.loads(event.get('body', '{}'))
            image_data = np.array(body['image']).reshape(784, 1) / 255.0

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, image_data)
            prediction = int(get_predictions(A2)[0])

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': json.dumps({'prediction': prediction})
            }
        elif event.get('httpMethod') == 'POST' and event.get('path') == '/api/train':
            body = json.loads(event.get('body', '{}'))
            image_data = np.array(body['image']).reshape(784, 1) / 255.0
            label = int(body['label'])

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, image_data)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, image_data, np.array([label]))
            global W1, b1, W2, b2
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, 0.1)

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': json.dumps({'status': 'Model trained on one sample'})
            }
        elif event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': ''
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

# This is the entry point for Vercel
def run(event, context):
    return handler(event, context)

# This is for local testing
if __name__ == "__main__":
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class MockHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            event = {'body': post_data.decode('utf-8'), 'path': self.path, 'httpMethod': 'POST'}
            response = handler(event, None)

            self.send_response(response['statusCode'])
            for key, value in response['headers'].items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(response['body'].encode('utf-8'))

    server = HTTPServer(('localhost', 8000), MockHandler)
    print('Starting server on http://localhost:8000')
    server.serve_forever()