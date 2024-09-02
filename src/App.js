import React, { useRef, useState, useEffect } from 'react';

const App = () => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.fillStyle = 'white';
    context.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.beginPath();
    context.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.lineWidth = 15;
    context.lineCap = 'round';
    context.strokeStyle = 'black';
    context.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    context.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.fillStyle = 'white';
    context.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setError(null);
  };

  const getImageData = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    const resizedData = new Array(28 * 28);
    const stepSize = canvas.width / 28;
  
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const sourcePosX = Math.floor(x * stepSize);
        const sourcePosY = Math.floor(y * stepSize);
        const index = (sourcePosY * canvas.width + sourcePosX) * 4;
        const avg = (imageData.data[index] + imageData.data[index + 1] + imageData.data[index + 2]) / 3;
        resizedData[y * 28 + x] = avg / 255; // Normalize to 0-1
      }
    }
    return resizedData;
  };  

  const handlePredict = async () => {
    const imageData = getImageData();
    console.log("Image data prepared:", imageData.length, "pixels");
    setError(null);
    
    try {
      console.log("Sending prediction request to:", '/api/predict');
      console.log("Request payload:", JSON.stringify({ image: imageData }));
      
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData }),
      });
      
      console.log("Received response:", response);
      console.log("Response status:", response.status);
      console.log("Response headers:", JSON.stringify([...response.headers]));
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error("Error response body:", errorText);
        throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
      }    
      
      const result = await response.json();
      console.log("Prediction result:", result);
      setPrediction(result.prediction);
    } catch (error) {
      console.error("Prediction error:", error);
      console.error("Error stack:", error.stack);
      setError(`Error making prediction: ${error.message}`);
    }    
  };

  const handleTrain = async () => {
    if (prediction === null) {
      setError('Please predict first before training');
      return;
    }
    const imageData = getImageData();
    setError(null);
    try {
      const response = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData, label: prediction }),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
      }
      const result = await response.json();
      alert(result.status);
      clearCanvas();
    } catch (error) {
      console.error("Training error:", error);
      setError(`Error training the model: ${error.message}`);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold mb-4">Digit Recognizer</h1>
      <div className="flex items-center space-x-8">
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
        {prediction !== null && (
          <div className="flex flex-col items-center">
            <div className="text-6xl font-bold text-blue-600 mb-4">
              {prediction}
            </div>
            <div className="text-xl">Predicted Digit</div>
          </div>
        )}
      </div>
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
      {error && (
        <div className="mt-4 text-red-500">
          {error}
        </div>
      )}
    </div>
  );
};

export default App;