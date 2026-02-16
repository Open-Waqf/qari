import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

const srcDir = path.join(root, "node_modules", "@tensorflow", "tfjs-backend-wasm", "dist");
const outDir = path.join(root, "public", "tfjs-wasm");

await fs.mkdir(outDir, {recursive: true });

const files = await fs.readdir(srcDir);
const wanted = files.filter(
    (f) => f.startsWith("tfjs-backend-wasm") && (f.endsWith(".wasm") || f.endsWith(".js"))
);

await Promise.all(
    wanted.map((f) => fs.copyFile(path.join(srcDir, f), path.join(outDir, f)))
);

console.log(`[tfjs-wasm] Copied ${wanted.length} files to public/tfjs-wasm/`);
