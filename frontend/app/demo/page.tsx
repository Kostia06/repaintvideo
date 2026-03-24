"use client"

import { useState } from "react"
import Link from "next/link"
import { WebcamFeed } from "@/components/WebcamFeed"
import { StylePicker } from "@/components/StylePicker"

export default function DemoPage() {
  const [selectedStyle, setSelectedStyle] = useState("monet")

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
          <StylePicker selected={selectedStyle} onChange={setSelectedStyle} />
          <p className="text-sm text-gray-400">
            Frames are styled in near real-time. Switch styles to compare.
          </p>
        </div>
      </div>
    </div>
  )
}
