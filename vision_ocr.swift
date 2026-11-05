import Foundation
import Vision
import AppKit

// Apple Vision OCR. Args: image paths. Output per line: <path>\t<recognized text>
let paths = Array(CommandLine.arguments.dropFirst())
for path in paths {
    guard let data = FileManager.default.contents(atPath: path),
          let img = NSImage(data: data),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("\(path)\t")
        continue
    }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = true
    req.recognitionLanguages = ["en-US"]
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    try? handler.perform([req])
    // Vision doesn't promise reading order, so a two-line subtitle could come
    // back bottom line first. Sort top-to-bottom (boundingBox origin is
    // bottom-left, so higher midY = higher on screen), left-to-right within a
    // line: boxes whose vertical centres are within half a line height match.
    let obs = (req.results ?? []).sorted { a, b in
        let ha = a.boundingBox.height, hb = b.boundingBox.height
        if abs(a.boundingBox.midY - b.boundingBox.midY) < min(ha, hb) / 2 {
            return a.boundingBox.minX < b.boundingBox.minX
        }
        return a.boundingBox.midY > b.boundingBox.midY
    }
    var lines: [String] = []
    for o in obs {
        if let c = o.topCandidates(1).first { lines.append(c.string) }
    }
    let text = lines.joined(separator: " ")
    print("\(path)\t\(text)")
}
