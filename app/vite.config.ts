import {defineConfig} from 'vite';
import {VitePWA} from 'vite-plugin-pwa';
import pkg from './package.json';
import {resolve} from 'path'

export default defineConfig({
    define: {
        '__APP_VERSION__': JSON.stringify(pkg.version),
    },
    build: {
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'index.html'),
                ar: resolve(__dirname, 'ar/index.html'),
            },
        },
    },
    plugins: [
        VitePWA({
            injectRegister: null,        // IMPORTANT: manual registration
            registerType: 'prompt',      // safer UX than auto reload
            includeAssets: [
                'pwa/favicon.ico',
                'pwa/apple-icon-180.png',
                'pwa/masked-icon.png',
                'og-image.jpg',
            ],
            manifest: false,
            workbox: {
                cleanupOutdatedCaches: true,
                // This is the "Magic" part for AI apps
                globPatterns: ['**/*.{js,css,html,json,woff2,ttf,wasm}'],
                maximumFileSizeToCacheInBytes: 50 * 1024 * 1024,
                runtimeCaching: [
                    {
                        urlPattern: ({url}) => url.pathname.startsWith('/models/'),
                        handler: 'CacheFirst',
                        options: {
                            cacheName: 'models-cache',
                            expiration: {maxEntries: 30, maxAgeSeconds: 60 * 60 * 24 * 30},
                        },
                    },
                ],
            },
        })
    ]
});