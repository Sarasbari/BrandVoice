import React from 'react';

const CONTENT_TYPES = [
  { label: 'Tweet', value: 'tweet' },
  { label: 'LinkedIn', value: 'linkedin' },
  { label: 'Blog Intro', value: 'blog_intro' },
  { label: 'Changelog', value: 'changelog' },
  { label: 'How-To', value: 'how_to' }
];

export default function ContentTypeSelector({ selected, onChange }) {
  return (
    <div className="w-full flex flex-col gap-2.5">
      <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        1. Select Content Format
      </span>
      <div className="flex flex-wrap gap-2.5">
        {CONTENT_TYPES.map((type) => {
          const isSelected = selected === type.value;
          return (
            <button
              key={type.value}
              type="button"
              id={`content-type-btn-${type.value}`}
              onClick={() => onChange(type.value)}
              className={`
                px-5 py-2.5 rounded-full text-sm font-medium tracking-tight
                transition-all duration-200 ease-out cursor-pointer
                active:scale-95 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-black
                ${isSelected 
                  ? 'bg-black text-white border-transparent shadow-sm' 
                  : 'bg-white text-gray-700 border border-gray-300 hover:border-black hover:text-black hover:bg-gray-50'
                }
              `}
            >
              {type.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
