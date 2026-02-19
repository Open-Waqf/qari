import {mkdirSync, readFileSync, writeFileSync} from 'node:fs'
import {dirname, resolve} from 'node:path'
import {type Lang, SEO, SITE} from '../src/core/seo'

const root = resolve(process.cwd())
const templatePath = resolve(root, 'index.template.html')
const template = readFileSync(templatePath, 'utf-8')

function ensureDir(filePath: string) {
    mkdirSync(dirname(filePath), {recursive: true})
}

function rep(s: string, search: string, replacement: string) {
    return s.split(search).join(replacement)
}

function buildHreflang() {
    return [
        `<link rel="alternate" hreflang="en" href="${SITE.origin}/" />`,
        `<link rel="alternate" hreflang="ar" href="${SITE.origin}/ar/" />`,
        `<link rel="alternate" hreflang="x-default" href="${SITE.origin}/" />`,
    ].join('\n  ')
}

function renderIndex(lang: Lang) {
    const s = SEO[lang]
    const canon = `${SITE.origin}${s.path}`
    const ogUrl = canon
    const manifestHref = lang === 'en' ? '/manifest.webmanifest' : '/manifest.ar.webmanifest'

    let out = template
    out = rep(out, '__LANG__', s.lang)
    out = rep(out, '__DIR__', s.dir)
    out = rep(out, '__TITLE__', s.title)
    out = rep(out, '__DESC__', s.description)
    out = rep(out, '__CANON__', canon)
    out = rep(out, '__HREFLANG__', buildHreflang())
    out = rep(out, '__OG_URL__', ogUrl)
    out = rep(out, '__OG_TITLE__', s.ogTitle)
    out = rep(out, '__OG_DESC__', s.ogDescription)
    out = rep(out, '__OG_IMAGE__', SITE.ogImage)
    out = rep(out, '__TW_TITLE__', s.twitterTitle)
    out = rep(out, '__TW_DESC__', s.twitterDescription)
    out = rep(out, '__MANIFEST__', manifestHref)
    return out;
}

function renderManifest(lang: Lang) {
    const s = SEO[lang]
    return {
        lang: s.lang,
        dir: s.dir,
        name: s.manifest.name,
        short_name: s.manifest.shortName,
        description: s.manifest.description,
        start_url: s.path,
        scope: '/',
        display: 'standalone',
        theme_color: '#0077ff',
        background_color: '#02040a',
        icons: [
            {src: 'pwa/pwa-192x192.png', sizes: '192x192', type: 'image/png'},
            {src: 'pwa/pwa-512x512.png', sizes: '512x512', type: 'image/png'},
            // if you have a maskable icon file, add it here too:
            // { src: 'pwa/maskable-icon-512x512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
    }
}

function write() {
    // EN index
    const enOut = resolve(root, 'index.html')
    writeFileSync(enOut, renderIndex('en'), 'utf-8')

    // AR index
    const arOut = resolve(root, 'ar/index.html')
    ensureDir(arOut)
    writeFileSync(arOut, renderIndex('ar'), 'utf-8')

    // manifests into /public
    const enManifestOut = resolve(root, 'public/manifest.webmanifest')
    ensureDir(enManifestOut)
    writeFileSync(enManifestOut, JSON.stringify(renderManifest('en'), null, 2), 'utf-8')

    const arManifestOut = resolve(root, 'public/manifest.ar.webmanifest')
    writeFileSync(arManifestOut, JSON.stringify(renderManifest('ar'), null, 2), 'utf-8')
}

write()
console.log('✅ Generated index.html, ar/index.html, manifest.webmanifest, manifest.ar.webmanifest')