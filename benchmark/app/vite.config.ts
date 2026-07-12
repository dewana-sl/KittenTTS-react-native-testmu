import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const dirname = path.dirname(fileURLToPath(import.meta.url));
const sdkLibPattern = new RegExp(
  `${path
    .resolve(dirname, '..', '..', 'lib')
    .replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\/.*`,
);

export default defineConfig(({mode}) => ({
  plugins: [react()],
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
