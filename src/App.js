import React, { useRef, useState, useEffect } from 'react';

const App = () => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.fillStyle = 'white';
    context.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.beginPath();
    context.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    context.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.fillStyle = 'white';
    context.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
  };

  const getImageData = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    return Array.from(imageData.data).filter((_, i) => i % 4 === 0);
  };

  const handlePredict = async () => {
    const imageData = getImageData();
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData }),
    });
    const result = await response.json();
    setPrediction(result.prediction);
  };

  const handleTrain = async () => {
    if (prediction === null) {
      alert('Please predict first before training');
      return;
    }
    const imageData = getImageData();
    const response = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData, label: prediction }),
    });
    const result = await response.json();
    alert(result.status);
    clearCanvas();
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold mb-4">Digit Recognizer</h1>
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseOut={stopDrawing}
        className="border-2 border-gray-300 rounded-lg shadow-md bg-white"
      />
      <div className="mt-4 space-x-4">
        <button
          onClick={handlePredict}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          Predict
        </button>
        <button
          onClick={handleTrain}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors"
        >
          Train
        </button>
        <button
          onClick={clearCanvas}
          className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
        >
          Clear
        </button>
      </div>
      {prediction !== null && (
        <div className="mt-4 text-xl font-bold">
          Prediction: {prediction}
        </div>
      )}
    </div>
  );
};

export default App;