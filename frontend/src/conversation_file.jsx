import React, { useState } from 'react';
import { Upload, Send, Loader2, Database, MessageCircle, FileCheck, AlertCircle } from 'lucide-react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

const API_BASE = "http://localhost:8000";

const GraphRAGUI = () => {
  const [chat, setChat] = useState([]);
  const [input, setInput] = useState("");
  const [uploadStatus, setUploadStatus] = useState("idle"); 
  const [isQuerying, setIsQuerying] = useState(false);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    setUploadStatus("loading");

    try {
      await axios.post(`${API_BASE}/upload`, formData);
      setUploadStatus("success");
      setChat(prev => [...prev, { role: 'system', content: `Success: ${file.name} is now part of the knowledge graph.` }]);
    } catch (err) {
      setUploadStatus("error");
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { role: 'user', content: input };
    setChat([...chat, userMsg]);
    setInput("");
    setIsQuerying(true);

    try {
      const response = await axios.post(`${API_BASE}/query`, { query: input });
      setChat(prev => [...prev, { role: 'ai', content: response.data.response }]);
    } catch (err) {
      setChat(prev => [...prev, { role: 'ai', content: "Connection error. Please check if backend is running." }]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="container-fluid vh-100 p-0 bg-light d-flex flex-column">
      
      {/* HEADER */}
      <nav className="navbar navbar-dark bg-dark px-4 shadow-sm">
        <span className="navbar-brand mb-0 h1 d-flex align-items-center gap-2">
          <Database className="text-info" /> GraphRAG Intelligence Studio
        </span>
      </nav>

      <div className="row g-0 flex-grow-1 overflow-hidden">
        
        {/* LEFT PANEL: UPLOAD & SOURCE MANAGEMENT */}
        <div className="col-md-4 col-lg-3 bg-white border-end d-flex flex-column p-4">
          <div className="mb-4">
            <h5 className="fw-bold text-secondary text-uppercase small">Knowledge Source</h5>
            <p className="text-muted small">Upload documents to build your private knowledge graph.</p>
          </div>

          <div className={`card border-2 border-dashed p-4 text-center transition-all 
            ${uploadStatus === 'loading' ? 'bg-light border-primary' : 'border-secondary-subtle'}`}>
            <input type="file" id="fileUpload" className="visually-hidden" onChange={handleFileUpload} />
            <label htmlFor="fileUpload" className="stretched-link cursor-pointer">
              {uploadStatus === 'loading' ? (
                <Loader2 className="text-primary spin mb-2" size={40} />
              ) : (
                <Upload className="text-secondary mb-2" size={40} />
              )}
              <div className="fw-bold">
                {uploadStatus === 'loading' ? "Processing..." : "Drop file here"}
              </div>
              <small className="text-muted text-decoration-underline">or click to browse</small>
            </label>
          </div>

          {uploadStatus === 'success' && (
            <div className="alert alert-success mt-3 d-flex align-items-center gap-2">
              <FileCheck size={18} /> Indexed Successfully
            </div>
          )}

          {uploadStatus === 'error' && (
            <div className="alert alert-danger mt-3 d-flex align-items-center gap-2">
              <AlertCircle size={18} /> Upload Failed
            </div>
          )}

          <div className="mt-auto">
            <div className="p-3 bg-light rounded-3 text-muted small">
              <strong>Tip:</strong> GraphRAG combines vector search with graph relationships for more accurate answers.
            </div>
          </div>
        </div>

        {/* RIGHT PANEL: CONVERSATION AREA */}
        <div className="col-md-8 col-lg-9 d-flex flex-column position-relative bg-white">
          
          {/* Messages Window */}
          <div className="flex-grow-1 overflow-y-auto p-4" style={{ backgroundColor: '#f8f9fa' }}>
            <div className="container-md">
              {chat.length === 0 && (
                <div className="text-center mt-5">
                  <MessageCircle size={60} className="text-light-emphasis opacity-25 mb-3" />
                  <h4 className="text-muted">Start a Conversation</h4>
                  <p className="text-secondary">Ask questions about your uploaded PDFs or Markdown files.</p>
                </div>
              )}

              {chat.map((msg, idx) => (
                <div key={idx} className={`d-flex mb-4 ${msg.role === 'user' ? 'justify-content-end' : 'justify-content-start'}`}>
                  <div className={`shadow-sm p-3 rounded-4 px-4 mw-75 
                    ${msg.role === 'user' ? 'bg-primary text-white rounded-bottom-end-0' : 'bg-white text-dark border rounded-bottom-start-0'}`}>
                    <div className="small opacity-75 mb-1 fw-bold">
                      {msg.role === 'user' ? 'YOU' : 'AI ASSISTANT'}
                    </div>
                    {msg.content}
                  </div>
                </div>
              ))}
              
              {isQuerying && (
                <div className="d-flex justify-content-start mb-4">
                  <div className="bg-white border p-3 rounded-4 shadow-sm">
                    <div className="spinner-grow spinner-grow-sm text-primary me-2" role="status"></div>
                    <span className="small text-muted italic">Thinking...</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Input Bar */}
          <div className="p-4 bg-white border-top shadow-lg">
            <div className="container-md">
              <div className="input-group input-group-lg shadow-sm rounded-pill overflow-hidden border">
                <input 
                  type="text" 
                  className="form-control border-0 px-4 bg-white" 
                  placeholder="Ask your data anything..." 
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                />
                <button 
                  className="btn btn-primary px-4 d-flex align-items-center" 
                  onClick={handleSendMessage}
                  disabled={isQuerying || !input.trim()}
                >
                  <Send size={20} />
                </button>
              </div>
            </div>
          </div>

        </div>
      </div>

      <style>{`
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .mw-75 { max-width: 75%; }
        .cursor-pointer { cursor: pointer; }
        .transition-all { transition: all 0.3s ease; }
        .border-dashed { border-style: dashed !important; }
        .rounded-bottom-end-0 { border-bottom-right-radius: 0 !important; }
        .rounded-bottom-start-0 { border-bottom-left-radius: 0 !important; }
      `}</style>
    </div>
  );
};

export default GraphRAGUI;