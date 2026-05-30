import React, { useState } from 'react';

export default function OutputPanel({ output, loading }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!output) return;
    navigator.clipboard.writeText(output)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch((err) => {
        console.error('Failed to copy text: ', err);
      });
  };

  return (
    <div className="w-full flex flex-col gap-2.5">
      <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        3. Generated Copy
      </span>

      {/* Empty State */}
      {!loading && !output && (
        <div 
          id="output-empty-state"
          className="w-full py-12 px-6 border-2 border-dashed border-gray-200 rounded-2xl flex flex-col items-center justify-center text-center bg-gray-50/50"
        >
          <svg className="w-8 h-8 text-gray-300 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19.5 12h-15m0 0l6.75 6.75M4.5 12l6.75-6.75" />
          </svg>
          <p className="text-sm text-gray-400 font-medium">
            Your generated content will appear here
          </p>
        </div>
      )}

      {/* Loading State: Pulsing Skeleton */}
      {loading && (
        <div 
          id="output-skeleton"
          className="w-full p-6 border border-gray-200 rounded-2xl bg-white shadow-sm flex flex-col gap-3.5"
        >
          <div className="h-4 bg-gray-200 rounded animate-pulse w-full"></div>
          <div className="h-4 bg-gray-200 rounded animate-pulse w-11/12"></div>
          <div className="h-4 bg-gray-200 rounded animate-pulse w-2/3"></div>
        </div>
      )}

      {/* Output State */}
      {!loading && output && (
        <div 
          id="output-content-panel"
          className="relative w-full p-6 border border-gray-200 rounded-2xl bg-white shadow-[0_2px_8px_rgba(0,0,0,0.02)] group hover:border-gray-300 transition-all duration-200"
        >
          {/* Copy Button */}
          <button
            id="copy-to-clipboard-btn"
            type="button"
            onClick={handleCopy}
            className={`
              absolute top-4 right-4 px-3 py-1.5 rounded-lg text-xs font-semibold
              transition-all duration-250 cursor-pointer flex items-center gap-1.5 border
              ${copied 
                ? 'bg-emerald-50 border-emerald-200 text-emerald-700' 
                : 'bg-white hover:bg-gray-50 border-gray-200 text-gray-600 hover:text-black hover:border-gray-300'
              }
            `}
          >
            {copied ? (
              <>
                <svg className="w-3.5 h-3.5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                </svg>
                Copied!
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                </svg>
                Copy
              </>
            )}
          </button>

          {/* Generated Text */}
          <div 
            id="output-generated-text"
            className="text-gray-800 text-sm leading-relaxed whitespace-pre-wrap pr-16 font-normal font-sans"
          >
            {output}
          </div>
        </div>
      )}
    </div>
  );
}
