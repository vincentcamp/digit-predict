from http.server import BaseHTTPRequestHandler
import json
import numpy as np

# Load the model parameters
with open('./digit_recognizer_model.json', 'r') as f:
    model_params = json.load(f)

W1 = np.array(model_params['W1'])
b1 = np.array(model_params['b1'])
W2 = np.array(model_params['W2'])
b2 = np.array(model_params['b2'])

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
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
    m = 1
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        if self.path == '/api/predict':
            image_data = np.array(data['image']).reshape(784, 1) / 255.0
            _, _, _, A2 = forward_prop(W1, b1, W2, b2, image_data)
            prediction = int(get_predictions(A2)[0])

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'prediction': prediction}).encode('utf-8'))

        elif self.path == '/api/train':
            global W1, b1, W2, b2
            image_data = np.array(data['image']).reshape(784, 1) / 255.0
            label = data['label']
            
            # Perform one step of training
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, image_data)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, image_data, label)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, 0.1)
            
            # Save updated model
            updated_model_params = {
                'W1': W1.tolist(),
                'b1': b1.tolist(),
                'W2': W2.tolist(),
                'b2': b2.tolist()
            }
            with open('./digit_recognizer_model.json', 'w') as f:
                json.dump(updated_model_params, f)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'Model updated successfully'}).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write('Hello, World!'.encode('utf-8'))