import {defineConfig} from 'vite';
import {VitePWA} from 'vite-plugin-pwa';

export default defineConfig({
    plugins: [
        VitePWA({
            registerType: 'autoUpdate',
            includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'masked-icon.svg'],
            manifest: {
                name: 'Qari Finder',
                short_name: 'QariFinder',
                description: 'Neural Voice Identification for Quran Reciters',
                theme_color: '#0077ff',
                background_color: '#02040a',
                display: 'standalone',
                icons: [
                    {
                        src: 'pwa-192x192.png',
                        sizes: '192x192',
                        type: 'image/png'
                    },
                    {
                        src: 'pwa-512x512.png',
                        sizes: '512x512',
                        type: 'image/png'
                    }
                ]
            },
            workbox: {
                // This is the "Magic" part for AI apps
                globPatterns: ['**/*.{js,css,html,json,bin,wav,woff2,ttf}'],
                maximumFileSizeToCacheInBytes: 50 * 1024 * 1024
            }
        })
    ]
});