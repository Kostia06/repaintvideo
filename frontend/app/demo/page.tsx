"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { WebcamFeed } from "@/components/WebcamFeed"
import { StylePicker, type StyleOption } from "@/components/StylePicker"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? ""

export default function DemoPage() {
  const [selectedStyle, setSelectedStyle] = useState("monet")
  const [styles, setStyles] = useState<StyleOption[]>([])

  useEffect(() => {
    fetch(`${API_BASE}/api/styles`)
      .then((res) => res.json())
      .then((data) => {
        setStyles(data.styles)
        const first = data.styles.find((s: StyleOption) => s.available)
        if (first) setSelectedStyle(first.key)
      })
      .catch(() => {})
  }, [])

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <Link
        href="/"
        className="text-sm text-gray-400 hover:text-white transition-colors mb-6 inline-block"
      >
        &larr; Back to home
      </Link>

      <h1 className="text-3xl font-bold text-white mb-8">Live Demo</h1>

      <div className="grid lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <WebcamFeed style={selectedStyle} />
        </div>
        <div className="space-y-4">
          <StylePicker selected={selectedStyle} onChange={setSelectedStyle} styles={styles} />
          <p className="text-sm text-gray-400">
            Frames are styled in near real-time. Switch styles to compare.
          </p>
        </div>
      </div>
    </div>
  )
}
