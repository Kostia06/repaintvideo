"use client"

import { clsx } from "clsx"

interface StyleOption {
  key: string
  name: string
  description: string
  accent: string
}

const STYLES: StyleOption[] = [
  { key: "monet", name: "Monet", description: "Soft impressionist brushwork", accent: "bg-blue-400" },
  { key: "starry_night", name: "Starry Night", description: "Swirling post-impressionist sky", accent: "bg-indigo-500" },
  { key: "cyberpunk", name: "Cyberpunk", description: "Neon-lit futuristic glow", accent: "bg-pink-500" },
  { key: "ukiyo_e", name: "Ukiyo-e", description: "Japanese woodblock elegance", accent: "bg-red-400" },
]

interface StylePickerProps {
  selected: string
  onChange: (style: string) => void
}

export function StylePicker({ selected, onChange }: StylePickerProps) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {STYLES.map((style) => (
        <button
          key={style.key}
          onClick={() => onChange(style.key)}
          className={clsx(
            "card p-4 text-left transition-all cursor-pointer",
            selected === style.key
              ? "ring-2 ring-brand border-brand"
              : "hover:border-white/20"
          )}
        >
          <div className={clsx("w-full h-20 rounded-lg mb-3 flex items-center justify-center", style.accent)}>
            <span className="text-white font-bold text-sm">{style.name}</span>
          </div>
          <h3 className="font-semibold text-white text-sm">{style.name}</h3>
          <p className="text-xs text-gray-400 mt-1">{style.description}</p>
        </button>
      ))}
    </div>
  )
}
