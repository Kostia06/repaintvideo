"use client"

import { useCallback, useRef, useState } from "react"
import { clsx } from "clsx"

interface VideoUploaderProps {
  onFile: (file: File) => void
  accept?: string
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function VideoUploader({ onFile, accept = ".mp4,.mov,.avi" }: VideoUploaderProps) {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (f: File) => {
      setFile(f)
      onFile(f)
    },
    [onFile],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const dropped = e.dataTransfer.files[0]
      if (dropped) handleFile(dropped)
    },
    [handleFile],
  )

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = e.target.files?.[0]
      if (selected) handleFile(selected)
    },
    [handleFile],
  )

  const clear = useCallback(() => {
    setFile(null)
    if (inputRef.current) inputRef.current.value = ""
  }, [])

  return (
    <div
      onClick={() => !file && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={clsx(
        "card p-8 text-center transition-all",
        !file && "cursor-pointer",
        isDragging
          ? "border-brand border-solid bg-brand/5"
          : "border-dashed border-white/20",
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleChange}
        className="hidden"
      />

      {file ? (
        <div className="flex items-center justify-between">
          <div className="text-left">
            <p className="text-white font-medium">{file.name}</p>
            <p className="text-sm text-gray-400">{formatSize(file.size)}</p>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation()
              clear()
            }}
            className="text-sm text-red-400 hover:text-red-300 transition-colors"
          >
            Remove
          </button>
        </div>
      ) : (
        <div>
          <p className="text-gray-300 mb-1">Drop your video here or click to browse</p>
          <p className="text-xs text-gray-500">MP4, MOV, or AVI</p>
        </div>
      )}
    </div>
  )
}
