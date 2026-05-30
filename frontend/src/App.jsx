import React, { useState, useEffect } from 'react';
import ContentTypeSelector from './components/ContentTypeSelector';
import PromptInput from './components/PromptInput';
import OutputPanel from './components/OutputPanel';
import { generateContent, checkHealth } from './api';

export default function App() {
  const [selectedType, setSelectedType] = useState('tweet');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking'); // checking | online | offline
  const [modelId, setModelId] = useState('');

  // Run a health check on mount to verify server connection
  useEffect(() => {
    async function verifyBackend() {
      try {
        const data = await checkHealth();
        if (data.status === 'ok') {
          setApiStatus('online');
          setModelId(data.model);
        } else {
          setApiStatus('offline');
        }
      } catch (err) {
        setApiStatus('offline');
      }
    }
    verifyBackend();
  }, []);

  const handleGenerate = async (inputs) => {
    setLoading(true);
    setError('');
    setOutput('');

    try {
      const payload = {
        content_type: selectedType,
        ...inputs
      };
      
      const result = await generateContent(payload);
      
      if (result && result.generated_text) {
        setOutput(result.generated_text);
      } else {
        setError('Received an empty response from the server.');
      }
    } catch (err) {
      console.error(err);
      setError(err.message || 'An unexpected error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen py-16 px-4 sm:px-6 lg:px-8 bg-[#fafafa]">
      <div className="max-w-2xl mx-auto flex flex-col gap-8">
        
        {/* Header */}
        <header className="flex flex-col gap-2 pb-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold tracking-tight text-black font-display">
              BrandVoice
            </h1>
            
            {/* Elegant API Status Indicator */}
            <div className="flex items-center gap-2">
              <span className={`inline-block w-2.5 h-2.5 rounded-full ${
                apiStatus === 'online' ? 'bg-emerald-500 animate-pulse' :
                apiStatus === 'offline' ? 'bg-rose-500' : 'bg-amber-500'
              }`} />
              <span className="text-xs font-semibold text-gray-500 capitalize">
                {apiStatus === 'online' ? 'Connected' :
                 apiStatus === 'offline' ? 'Offline' : 'Checking connection...'}
              </span>
            </div>
          </div>
          <p className="text-sm text-gray-500 font-medium tracking-tight">
            Notion Brand Voice Content Generator
          </p>
        </header>

        {/* Global Error Banner */}
        {error && (
          <div className="p-4 bg-rose-50 border border-rose-100 rounded-xl flex items-start gap-3">
            <svg className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div className="flex flex-col gap-1">
              <span className="text-sm font-semibold text-rose-800">Generation Error</span>
              <p className="text-xs text-rose-700 leading-normal">{error}</p>
            </div>
          </div>
        )}

        {/* 1. ContentTypeSelector */}
        <ContentTypeSelector 
          selected={selectedType} 
          onChange={(type) => {
            setSelectedType(type);
            // Optionally clear output when content type changes to avoid confusion
            setOutput('');
          }} 
        />

        {/* 2. PromptInput */}
        <PromptInput 
          onSubmit={handleGenerate} 
          loading={loading} 
        />

        {/* 3. OutputPanel */}
        <OutputPanel 
          output={output} 
          loading={loading} 
        />

        {/* Footer info card */}
        {apiStatus === 'online' && modelId && (
          <footer className="mt-4 p-4 rounded-xl bg-gray-50 border border-gray-150 text-center">
            <p className="text-xs text-gray-400 font-normal">
              Active adapters hosted on Hugging Face Hub: <code className="text-xs bg-gray-100 px-1.5 py-0.5 rounded font-mono text-gray-600">{modelId}</code>
            </p>
          </footer>
        )}

      </div>
    </div>
  );
}
