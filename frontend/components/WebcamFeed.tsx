"use client"

import { useEffect, useRef, useState } from "react"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? ""

interface WebcamFeedProps {
  style: string
}

export function WebcamFeed({ style }: WebcamFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [styledSrc, setStyledSrc] = useState<string | null>(null)
  const [status, setStatus] = useState<"loading" | "active" | "denied">("loading")
  const streamRef = useRef<MediaStream | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    let cancelled = false

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
        setStatus("active")
      } catch {
        setStatus("denied")
      }
    }

    startCamera()

    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  useEffect(() => {
    if (status !== "active") return

    async function captureAndStyle() {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas || video.readyState < 2) return

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext("2d")
      if (!ctx) return

      ctx.drawImage(video, 0, 0)

      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.8),
      )
      if (!blob) return

      try {
        const form = new FormData()
        form.append("file", blob, "frame.jpg")
        form.append("style", style)

        const res = await fetch(`${API_BASE}/api/style/image`, {
          method: "POST",
          body: form,
        })

        if (res.ok) {
          const resultBlob = await res.blob()
          const url = URL.createObjectURL(resultBlob)
          setStyledSrc((prev) => {
            if (prev) URL.revokeObjectURL(prev)
            return url
          })
        }
      } catch {
        // silently skip failed frames
      }
    }

    intervalRef.current = setInterval(captureAndStyle, 200)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [status, style])

  useEffect(() => {
    return () => {
      if (styledSrc) URL.revokeObjectURL(styledSrc)
    }
  }, [styledSrc])

  if (status === "loading") {
    return (
      <div className="card p-8 text-center">
        <p className="text-gray-300">Requesting camera access...</p>
      </div>
    )
  }

  if (status === "denied") {
    return (
      <div className="card p-8 text-center">
        <p className="text-red-400 font-medium">Camera not available</p>
        <p className="text-sm text-gray-400 mt-1">
          Please allow camera access and reload the page.
        </p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <p className="text-xs text-gray-400 mb-2">Live camera</p>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full rounded-lg bg-black"
        />
      </div>
      <div>
        <p className="text-xs text-gray-400 mb-2">Styled output</p>
        {styledSrc ? (
          <img src={styledSrc} alt="Styled frame" className="w-full rounded-lg" />
        ) : (
          <div className="w-full aspect-video rounded-lg bg-white/5 flex items-center justify-center">
            <p className="text-sm text-gray-500">Waiting for first frame...</p>
          </div>
        )}
      </div>
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}
