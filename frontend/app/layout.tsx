import type { Metadata } from "next"
import Link from "next/link"
import "./globals.css"

export const metadata: Metadata = {
  title: "RepaintVideo",
  description: "Neural style transfer for video",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <nav className="sticky top-0 z-50 flex items-center justify-between px-6 py-4 border-b border-white/10 bg-[#0a0a0a]/80 backdrop-blur-md">
          <Link href="/" className="text-xl font-bold tracking-tight text-white">
            RepaintVideo
          </Link>
          <Link
            href="/demo"
            className="text-sm font-medium text-brand hover:text-brand-dark transition-colors"
          >
            Live Demo
          </Link>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  )
}
