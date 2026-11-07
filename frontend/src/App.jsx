
import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(selectedFile.type)) {
      setError("Please choose a JPEG, PNG, or WebP image.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Please choose an image smaller than 10 MB.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  }

  async function handleAnalyze() {
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Image analysis failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the backend. Check that FastAPI is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
  }

  const isReal = result?.label === "REAL";

  return (
    <main className="app-shell">
      <nav className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">◈</span>
          Verity<span className="brand-light">AI</span>
        </a>
        <span className="status-pill">
          <span className="status-dot" />
          CNN detector
        </span>
      </nav>

      <section className="hero">
        <p className="eyebrow">IMAGE FORENSICS · BASELINE MODEL</p>
        <h1>
          Real or artificial?
          <br />
          <span>Look a little closer.</span>
        </h1>
        <p className="hero-copy">
          Upload a facial image and let our convolutional neural network
          analyze the visual patterns it has learned.
        </p>
      </section>

      <section className="workspace">
        <div className="upload-panel">
          <div className="panel-heading">
            <div>
              <p className="step-label">01 / INPUT</p>
              <h2>Upload an image</h2>
            </div>
            {file && (
              <button className="text-button" onClick={handleReset}>
                Remove
              </button>
            )}
          </div>

          <label className={`dropzone ${preview ? "has-image" : ""}`}>
            {preview ? (
              <img className="image-preview" src={preview} alt="Selected face" />
            ) : (
              <div className="upload-prompt">
                <span className="upload-icon">↑</span>
                <strong>Drop your image into the spotlight</strong>
                <span>or click to browse your files</span>
                <small>JPEG · PNG · WEBP · MAX 10 MB</small>
              </div>
            )}
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
            />
          </label>

          {file && <p className="file-name">{file.name}</p>}

          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={!file || loading}
          >
            {loading ? (
              <>
                <span className="spinner" />
                Analyzing image...
              </>
            ) : (
              <>Analyze image <span>↗</span></>
            )}
          </button>

          {error && <p className="error-message">{error}</p>}
        </div>

        <div className="result-panel">
          <p className="step-label">02 / ANALYSIS</p>
          <h2>Your result</h2>

          {!result && !loading && (
            <div className="empty-result">
              <div className="scan-icon">⌕</div>
              <strong>Awaiting an image</strong>
              <p>Your prediction will appear here after analysis.</p>
            </div>
          )}

          {loading && (
            <div className="empty-result">
              <span className="large-spinner" />
              <strong>Examining visual patterns</strong>
              <p>The CNN is processing your image.</p>
            </div>
          )}

          {result && (
            <div className="result-content">
              <div className={`verdict ${isReal ? "real" : "fake"}`}>
                <span className="verdict-symbol">{isReal ? "✓" : "!"}</span>
                <div>
                  <small>MODEL PREDICTION</small>
                  <h3>{isReal ? "Likely real" : "AI-generated"}</h3>
                </div>
              </div>

              <div className="confidence-heading">
                <span>Predicted-class probability</span>
                <strong>{(result.confidence * 100).toFixed(2)}%</strong>
              </div>
              <div className="confidence-track">
                <div
                  className={`confidence-fill ${isReal ? "real" : "fake"}`}
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>

              <div className="probability-row">
                <span>AI-generated</span>
                <strong>
                  {(result.probabilities.AI_GENERATED * 100).toFixed(2)}%
                </strong>
              </div>
              <div className="probability-row">
                <span>Real</span>
                <strong>
                  {(result.probabilities.REAL * 100).toFixed(2)}%
                </strong>
              </div>

              <p className="model-note">
                Model: {result.model} · 128 × 128 input
              </p>
              <p className="disclaimer">
                This is a model prediction, not proof of authenticity. The
                baseline was trained on real FFHQ and StyleGAN-generated faces.
                Results may differ for other image sources.
              </p>
            </div>
          )}
        </div>
      </section>

      <footer>
        <span>VERITY AI · RESEARCH PROTOTYPE</span>
        <span>Built with a PyTorch CNN</span>
      </footer>
    </main>
  );
}

export default App;
