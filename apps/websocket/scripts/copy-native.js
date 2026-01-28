import fs from "node:fs";
import path from "node:path";

const root = process.cwd();

const candidates = [
  {
    src: path.join(root, "native", "build"),
    dest: path.join(root, "dist", "native", "build"),
  },
  {
    src: path.join(root, "apps", "websocket", "native", "build"),
    dest: path.join(root, "apps", "websocket", "dist", "native", "build"),
  },
];

const target = candidates.find((entry) => fs.existsSync(entry.src));

if (!target) {
  console.warn("[copy-native] native/build not found; skipping copy.");
  process.exit(0);
}

fs.mkdirSync(path.dirname(target.dest), { recursive: true });
fs.rmSync(target.dest, { recursive: true, force: true });
fs.cpSync(target.src, target.dest, { recursive: true });

console.log(`[copy-native] Copied ${target.src} -> ${target.dest}`);
