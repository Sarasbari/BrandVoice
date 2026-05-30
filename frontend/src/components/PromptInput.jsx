import React, { useState } from 'react';

export default function PromptInput({ onSubmit, loading }) {
  const [topic, setTopic] = useState('');
  const [audience, setAudience] = useState('');
  const [tone, setTone] = useState('');
  const [error, setError] = useState('');

  const handleGenerate = () => {
    setError('');

    // Field validation
    if (!topic.trim() || !audience.trim() || !tone.trim()) {
      setError('All fields (Topic, Audience, and Tone) are required.');
      return;
    }

    onSubmit({
      topic: topic.trim(),
      audience: audience.trim(),
      tone: tone.trim()
    });
  };

  return (
    <div className="w-full flex flex-col gap-5 bg-white p-6 rounded-2xl border border-gray-150 shadow-[0_2px_8px_rgba(0,0,0,0.04)]">
      <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider -mb-2">
        2. Set Parameters
      </span>

      {error && (
        <div className="text-sm text-red-500 bg-red-50 border border-red-100 px-4 py-2.5 rounded-lg">
          {error}
        </div>
      )}

      <div className="flex flex-col gap-4">
        {/* Topic Input */}
        <div className="flex flex-col gap-1.5">
          <label htmlFor="input-topic" className="text-sm font-semibold text-gray-800">
            Topic <span className="text-red-500">*</span>
          </label>
          <input
            id="input-topic"
            type="text"
            placeholder="e.g., Launching formulas in tables"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            disabled={loading}
            className="w-full px-4 py-3 rounded-lg border border-gray-300 text-sm focus:outline-none focus:border-black focus:ring-1 focus:ring-black transition-all bg-gray-50 focus:bg-white disabled:opacity-60"
          />
        </div>

        {/* Audience Input */}
        <div className="flex flex-col gap-1.5">
          <label htmlFor="input-audience" className="text-sm font-semibold text-gray-800">
            Audience <span className="text-red-500">*</span>
          </label>
          <input
            id="input-audience"
            type="text"
            placeholder="e.g., Knowledge workers and startup founders"
            value={audience}
            onChange={(e) => setAudience(e.target.value)}
            disabled={loading}
            className="w-full px-4 py-3 rounded-lg border border-gray-300 text-sm focus:outline-none focus:border-black focus:ring-1 focus:ring-black transition-all bg-gray-50 focus:bg-white disabled:opacity-60"
          />
        </div>

        {/* Tone Input */}
        <div className="flex flex-col gap-1.5">
          <label htmlFor="input-tone" className="text-sm font-semibold text-gray-800">
            Tone <span className="text-red-500">*</span>
          </label>
          <input
            id="input-tone"
            type="text"
            placeholder="e.g., excited but grounded, helpful"
            value={tone}
            onChange={(e) => setTone(e.target.value)}
            disabled={loading}
            className="w-full px-4 py-3 rounded-lg border border-gray-300 text-sm focus:outline-none focus:border-black focus:ring-1 focus:ring-black transition-all bg-gray-50 focus:bg-white disabled:opacity-60"
          />
        </div>
      </div>

      {/* Submit Button */}
      <button
        id="submit-generate-btn"
        type="button"
        onClick={handleGenerate}
        disabled={loading}
        className="w-full mt-2 py-3.5 rounded-lg bg-black text-white font-medium text-sm tracking-tight transition-all duration-200 cursor-pointer hover:bg-neutral-800 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-sm"
      >
        {loading ? (
          <>
            <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Generating...
          </>
        ) : (
          'Generate Content'
        )}
      </button>
    </div>
  );
}
