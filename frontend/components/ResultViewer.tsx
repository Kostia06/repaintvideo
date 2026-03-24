"use client"

import { useEffect, useState } from "react"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? ""

interface JobStatus {
  status: string
  progress: number
  done: boolean
  error: string | null
}

interface ResultViewerProps {
  jobId: string | null
  isImage: boolean
  imageUrl?: string
}

export function ResultViewer({ jobId, isImage, imageUrl }: ResultViewerProps) {
  const [job, setJob] = useState<JobStatus | null>(null)

  useEffect(() => {
    if (!jobId || isImage) return

    setJob({ status: "queued", progress: 0, done: false, error: null })

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/job/${jobId}`)
        const data: JobStatus = await res.json()
        setJob(data)

        if (data.done || data.status === "error") {
          clearInterval(interval)
        }
      } catch {
        clearInterval(interval)
        setJob((prev) =>
          prev ? { ...prev, status: "error", error: "Connection lost" } : null,
        )
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [jobId, isImage])

  if (isImage && imageUrl) {
    return (
      <div className="card p-4">
        <img
          src={imageUrl}
          alt="Styled result"
          className="w-full rounded-lg"
        />
      </div>
    )
  }

  if (!jobId || !job) return null

  if (job.status === "error") {
    return (
      <div className="card p-6 text-center">
        <p className="text-red-400 font-medium">Processing failed</p>
        <p className="text-sm text-gray-400 mt-1">{job.error}</p>
      </div>
    )
  }

  if (job.done) {
    return (
      <div className="card p-6 text-center">
        <p className="text-brand font-medium mb-4">Video ready!</p>
        <a
          href={`${API_BASE}/api/download/${jobId}`}
          download
          className="inline-flex items-center gap-2 bg-brand hover:bg-brand-dark text-white font-medium px-6 py-3 rounded-lg transition-colors"
        >
          Download styled video
        </a>
      </div>
    )
  }

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm text-gray-300">
          {job.status === "queued" ? "Queued..." : "Processing..."}
        </p>
        <span className="text-sm text-gray-400">{job.progress}%</span>
      </div>
      <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full bg-brand rounded-full transition-all duration-300"
          style={{ width: `${job.progress}%` }}
        />
      </div>
    </div>
  )
}
