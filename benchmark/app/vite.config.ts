import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import {copyFile, mkdir} from 'node:fs/promises';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const sdkLibPattern = new RegExp(
  `${path
    .resolve(dirname, '..', '..', 'lib')
    .replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\/.*`,
);
const ortAssetFiles = [
  'ort.wasm.min.js',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
];

function copyOnnxRuntimeAssets() {
  return {
    name: 'copy-onnxruntime-web-assets',
    async closeBundle() {
      const ortDistDir = path.dirname(require.resolve('onnxruntime-web/wasm'));
      const outDir = path.resolve(dirname, 'dist', 'ort');

      await mkdir(outDir, {recursive: true});
      await Promise.all(
        ortAssetFiles.map(file =>
          copyFile(path.join(ortDistDir, file), path.join(outDir, file)),
        ),
      );
    },
  };
}

export default defineConfig(({mode}) => ({
  plugins: [react(), copyOnnxRuntimeAssets()],
  define: {
    __DEV__: JSON.stringify(mode !== 'production'),
  },
  resolve: {
    alias: [{find: /^react-native$/, replacement: 'react-native-web'}],
    conditions: ['browser', 'import', 'module', 'default'],
    extensions: [
      '.web.tsx',
      '.web.ts',
      '.tsx',
      '.ts',
      '.web.jsx',
      '.web.js',
      '.jsx',
      '.js',
      '.json',
    ],
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    commonjsOptions: {
      include: [/node_modules/, sdkLibPattern],
    },
  },
  server: {
    allowedHosts: ['localhost.lambdatest.com'],
    host: '0.0.0.0',
    port: 5173,
  },
  preview: {
    allowedHosts: ['localhost.lambdatest.com'],
    host: '0.0.0.0',
    port: 4173,
  },
  optimizeDeps: {
    esbuildOptions: {
      resolveExtensions: [
        '.web.tsx',
        '.web.ts',
        '.tsx',
        '.ts',
        '.web.jsx',
        '.web.js',
        '.jsx',
        '.js',
        '.json',
      ],
    },
  },
}));
