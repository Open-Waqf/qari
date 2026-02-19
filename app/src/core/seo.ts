export type Lang = 'en' | 'ar'

export const SITE = {
    origin: 'https://qari.open-waqf.org',
    ogImage: 'https://qari.open-waqf.org/og-image.jpg',
} as const

export const SEO: Record<Lang, {
    lang: Lang
    dir: 'ltr' | 'rtl'
    path: '/' | '/ar/'
    title: string
    description: string
    ogTitle: string
    ogDescription: string
    twitterTitle: string
    twitterDescription: string
    manifest: {
        name: string
        shortName: string
        description: string
    }
}> = {
    en: {
        lang: 'en',
        dir: 'ltr',
        path: '/',
        title: 'Qari Finder: Real-time Quran Reciter Identification',
        description: 'Identify world-famous Quran reciters instantly. 100% private, on-device AI.',
        ogTitle: 'Qari Finder: Real-time Quran ID',
        ogDescription: 'Identify your favorite Qaris instantly with local AI. No audio leaves your device.',
        twitterTitle: 'Qari Finder: Real-time Quran ID',
        twitterDescription: "Identify world-famous Quran reciters instantly. 100% private, on-device AI.",
        manifest: {
            name: 'Qari Finder',
            shortName: 'Qari Finder',
            description: 'Identify world-famous Quran reciters instantly. 100% private, on-device AI.',
        },
    },
    ar: {
        lang: 'ar',
        dir: 'rtl',
        path: '/ar/',
        title: 'قارئ فايندر: التعرف الفوري على قارئ القرآن',
        description: 'تعرّف على قارئك المفضل فوراً. ذكاء اصطناعي يعمل على جهازك مع خصوصية كاملة.',
        ogTitle: 'قارئ فايندر: التعرف الفوري على القرّاء',
        ogDescription: 'اعرف قارئك المفضل فوراً بالذكاء الاصطناعي المحلي. لا يغادر الصوت جهازك.',
        twitterTitle: 'قارئ فايندر: التعرف الفوري على القرّاء',
        twitterDescription: 'تعرّف على قارئك المفضل فوراً. ذكاء اصطناعي يعمل على جهازك مع خصوصية كاملة.',
        manifest: {
            name: 'قارئ فايندر',
            shortName: 'قارئ',
            description: 'التعرّف الذكي على قرّاء القرآن على جهازك',
        },
    },
}
