import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [model, setModel] = useState("1");
  const [prediction, setPrediction] = useState(null);
  const [prob, setProb] = useState(null);

  // Resource links about deepfakes, bots, catfishing, etc.
  const resources = [

  ];

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    // Create a preview URL so we can display the selected image
    if (selectedFile) {
      setPreviewUrl(URL.createObjectURL(selectedFile));
    } else {
      setPreviewUrl(null);
    }
    // Reset previous predictions
    setPrediction(null);
    setProb(null);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("No file selected.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        `http://localhost:8000/predict/${model}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setPrediction(res.data.prediction);
      setProb(res.data.probability);
    } catch (err) {
      console.error(err);
      alert("Error uploading or predicting.");
    }
  };

  // Decide color for the probability bar
  // Green if prob < 0.33, orange if 0.33-0.66, red if > 0.66
  let barColor = "#2ecc71"; // green
  if (prob !== null) {
    if (prob > 0.66) barColor = "#e74c3c"; // red
    else if (prob > 0.33) barColor = "#f39c12"; // orange
  }

  return (
    <div className="app-container">
      <h1>Fake Face Detector</h1>

      <div className="model-select">
        <label>Choose Model:</label>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
        <option value="1">Model 1 (Basic CNN)</option>
    <option value="2">Model 2 (Enhanced CNN)</option>
    <option value="3">Model 3 (SIFT+SVM)</option>
    <option value="4">Model 4 (Haar+AdaBoost)</option>
        </select>
      </div>

      <div className="upload-section">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button onClick={handleUpload}>Upload & Predict</button>
      </div>

      {/* Image Preview */}
      {previewUrl && (
        <div className="preview-container">
          <img src={previewUrl} alt="Preview" className="preview-image" />
        </div>
      )}

      {/* Results */}
      {prob !== null && (
        <div className="result-section">
          <div className="card">
            <h2>
              {prediction === "Fake"
                ? "This image is likely FAKE"
                : "This image is likely REAL"}
            </h2>
            <h3>Probability of Fake: {(prob * 100).toFixed(2)}%</h3>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${prob * 100}%`,
                  backgroundColor: barColor,
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Extra resources */}
      <div className="resources-section">

        <ul>
          {resources.map((r) => (
            <li key={r.url}>
              <a href={r.url} target="_blank" rel="noopener noreferrer">
                {r.name}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default App;
