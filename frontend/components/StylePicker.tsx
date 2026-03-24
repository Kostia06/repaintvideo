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
  { key: "anime", name: "Anime", description: "Cel-shaded animation style", accent: "bg-emerald-400" },
  { key: "watercolor", name: "Watercolor", description: "Soft wet-edge painting", accent: "bg-teal-400" },
  { key: "pixel_art", name: "Pixel Art", description: "Retro game aesthetic", accent: "bg-green-500" },
  { key: "oil_painting", name: "Oil Painting", description: "Thick impasto brushwork", accent: "bg-amber-500" },
  { key: "pop_art", name: "Pop Art", description: "Bold Warhol-style colors", accent: "bg-yellow-400" },
  { key: "sketch", name: "Sketch", description: "Pencil drawing effect", accent: "bg-gray-400" },
  { key: "vintage", name: "Vintage", description: "Retro film nostalgia", accent: "bg-orange-400" },
  { key: "neon_glow", name: "Neon Glow", description: "Glowing rainbow edges", accent: "bg-purple-500" },
]

interface StylePickerProps {
  selected: string
  onChange: (style: string) => void
}

export function StylePicker({ selected, onChange }: StylePickerProps) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
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
