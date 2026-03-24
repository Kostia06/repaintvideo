"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { StylePicker } from "@/components/StylePicker"
import { VideoUploader } from "@/components/VideoUploader"
import { ResultViewer } from "@/components/ResultViewer"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? ""

export default function HomePage() {
  const [selectedStyle, setSelectedStyle] = useState("monet")
  const [file, setFile] = useState<File | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [availableStyles, setAvailableStyles] = useState<string[]>([])

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((res) => res.json())
      .then((data) => {
        console.log("Available styles:", data.styles)
        setAvailableStyles(data.styles)
      })
      .catch(() => console.log("API not reachable — running in static mode"))
  }, [])

  async function handleProcess() {
    if (!file) return
    setIsProcessing(true)
    setJobId(null)

    try {
      const form = new FormData()
      form.append("file", file)
      form.append("style", selectedStyle)

      const res = await fetch(`${API_BASE}/api/style/video`, {
        method: "POST",
        body: form,
      })
      const data = await res.json()
      setJobId(data.job_id)
    } catch (err) {
      console.error("Failed to submit video:", err)
    } finally {
      setIsProcessing(false)
    }
  }

  const hasModels = availableStyles.length > 0

  return (
    <div className="max-w-5xl mx-auto px-4">
      {/* Hero */}
      <section className="py-20 text-center">
        <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-white mb-4">
          Paint your world
        </h1>
        <p className="text-lg text-gray-400 max-w-xl mx-auto mb-8">
          Transform any video into a living painting with neural style transfer
          and temporal consistency.
        </p>
        <div className="flex items-center justify-center gap-4">
          <a
            href="#upload"
            className="bg-brand hover:bg-brand-dark text-white font-medium px-6 py-3 rounded-lg transition-colors"
          >
            Upload a video
          </a>
          <Link
            href="/demo"
            className="text-brand hover:text-brand-dark font-medium px-6 py-3 transition-colors"
          >
            Try live demo &rarr;
          </Link>
        </div>
      </section>

      {/* Upload section */}
      <section id="upload" className="py-12 space-y-6">
        <h2 className="text-2xl font-bold text-white">Choose a style</h2>
        {!hasModels && (
          <div className="card p-4 border-yellow-500/30 bg-yellow-500/5">
            <p className="text-sm text-yellow-400">
              Models not loaded yet. Upload weights to your HF Hub repo to enable styling.
            </p>
          </div>
        )}
        <StylePicker selected={selectedStyle} onChange={setSelectedStyle} />

        <h2 className="text-2xl font-bold text-white pt-4">Upload your video</h2>
        <VideoUploader onFile={setFile} />

        <button
          onClick={handleProcess}
          disabled={!file || isProcessing}
          className="w-full bg-brand hover:bg-brand-dark disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-3 rounded-lg transition-colors"
        >
          {isProcessing ? "Submitting..." : "Process video"}
        </button>

        <ResultViewer jobId={jobId} isImage={false} />
      </section>

      {/* How it works */}
      <section className="py-16 border-t border-white/10">
        <h2 className="text-2xl font-bold text-white mb-8">How it works</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div>
            <h3 className="font-semibold text-white mb-2">Temporal consistency</h3>
            <p className="text-sm text-gray-400">
              Optical flow warping between frames ensures smooth, flicker-free
              transitions so the painted effect feels natural across the entire video.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-white mb-2">VGG-19 perceptual loss</h3>
            <p className="text-sm text-gray-400">
              A pre-trained VGG-19 network extracts content and style features at
              multiple layers, balancing fidelity to the original with the target
              painting aesthetic.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-white mb-2">Real-time ONNX inference</h3>
            <p className="text-sm text-gray-400">
              A lightweight feed-forward network trained per style runs via ONNX Runtime,
              delivering fast inference without needing a GPU at serving time.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 border-t border-white/10 text-center text-sm text-gray-500">
        RepaintVideo &middot; Built with PyTorch + Next.js &middot;{" "}
        <a
          href="https://github.com/YOUR_USERNAME/repaintvideo"
          className="text-gray-400 hover:text-white transition-colors"
        >
          Open source on GitHub
        </a>
      </footer>
    </div>
  )
}
