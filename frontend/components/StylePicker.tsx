"use client"

import { clsx } from "clsx"

export interface StyleOption {
  key: string
  label: string
  description: string
  available: boolean
}

interface StylePickerProps {
  selected: string
  onChange: (style: string) => void
  styles: StyleOption[]
}

export function StylePicker({ selected, onChange, styles }: StylePickerProps) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {styles.map((style) => (
        <button
          key={style.key}
          onClick={() => style.available && onChange(style.key)}
          disabled={!style.available}
          className={clsx(
            "card p-4 text-left transition-all",
            style.available ? "cursor-pointer" : "opacity-40 cursor-not-allowed",
            selected === style.key
              ? "ring-2 ring-brand border-brand"
              : "hover:border-white/20",
          )}
        >
          <h3 className="font-semibold text-white text-sm">{style.label}</h3>
          <p className="text-xs text-gray-400 mt-1">{style.description}</p>
          {!style.available && (
            <p className="text-xs text-red-400 mt-1">Weights not loaded</p>
          )}
        </button>
      ))}
    </div>
  )
}
