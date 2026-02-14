import {EVENTS} from "./events.ts";

export type Lang = 'en' | 'ar';

export const translations = {
    en: {
        // --- Header & General ---
        appTitle: "Qari",
        appAccent: "Finder",
        subtitle: "Neural Voice Identification",
        acousticLabel: "Acoustic Signature",
        install: "📥 Install App",

        // --- Control & States ---
        initBtn: "Initialize Engine",
        resume: "▶ Resume",
        paused: "Paused",
        loading: "Loading Brain...",
        retry: "Retry Connection",
        online: "🚀 System Online",

        // --- Status Pill States ---
        awaiting: "Awaiting Signal",
        analyzing: "Listening...",       // Used for "Scanning" ring
        stabilizing: "Verifying...",     // Used when score is > 60% but < 80%
        confirmed: "Match Found",        // Used when score > 80%
        tooQuiet: "Too Quiet",
        unknown: "Unsure",

        // --- Glass Card (Result) ---
        listenBtn: "Listen on YouTube",
        dismiss: "Dismiss",

        // --- Match History ---
        historyTitle: "Recent Matches",
        noMatches: "No matches yet...",

        // --- Onboarding Modal ---
        welcomeTitle: "Welcome to Qari Finder",
        welcomeDesc: "Identify Quran reciters instantly using on-device AI. No audio leaves your phone.",
        getStarted: "Start Calibration",
        calibTitle: "Calibrating...",
        calibDesc: "Measuring background noise levels. Please remain silent for a moment.",
        scienceTitle: "Privacy First",
        scienceDesc: "We use local neural networks to analyze frequency patterns strictly on your device.",
        back: "Back",

        // --- Errors ---
        micDenied: "❌ Mic Access Denied",
        missingFile: "❌ Model File Missing"
    },
    ar: {
        // --- العناوين والعام ---
        appTitle: "قارئ",
        appAccent: "فايندر",
        subtitle: "التعرف الذكي على القراء",
        acousticLabel: "البصمة الصوتية",
        install: "📥 تثبيت التطبيق",

        // --- التحكم والحالات ---
        initBtn: "تشغيل المحرك",
        resume: "▶ استئناف",
        paused: "متوقف مؤقتاً",
        loading: "جاري تحميل النموذج...",
        retry: "إعادة المحاولة",
        online: "🚀 النظام جاهز",

        // --- حالات شريط الحالة ---
        awaiting: "بانتظار الإشارة",
        analyzing: "جاري الاستماع...",
        stabilizing: "التحقق من التطابق...",
        confirmed: "تم التعرف عليه",
        tooQuiet: "الصوت منخفض",
        unknown: "غير مؤكد",

        // --- بطاقة النتيجة (Glass Card) ---
        listenBtn: "شاهد على يوتيوب",
        dismiss: "إغلاق",

        // --- سجل القراء (History) ---
        historyTitle: "آخر القراءات",
        noMatches: "لا توجد تطابقات بعد...",

        // --- نافذة الترحيب (Onboarding) ---
        welcomeTitle: "مرحباً بك في قارئ فايندر",
        welcomeDesc: "تعرف على القراء فورياً باستخدام الذكاء الاصطناعي المحلي. خصوصية تامة، الصوت لا يغادر هاتفك.",
        getStarted: "بدء المعايرة",
        calibTitle: "جاري المعايرة...",
        calibDesc: "نقوم بقياس مستوى الضجيج في الغرفة. يرجى التزام الصمت للحظة.",
        scienceTitle: "الخصوصية أولاً",
        scienceDesc: "نستخدم شبكات عصبية محلية لتحليل أنماط الترددات مباشرة على جهازك.",
        back: "رجوع",

        // --- الأخطاء ---
        micDenied: "❌ الميكروفون محظور",
        missingFile: "❌ ملف النموذج مفقود"
    }
};

class I18nManager {
    // Default to Arabic if you prefer, or detect browser language
    currentLang: Lang = 'ar';

    toggle(): Lang {
        this.currentLang = this.currentLang === 'en' ? 'ar' : 'en';
        this.apply();
        return this.currentLang;
    }

    constructor() {
        this.apply();
    }

    apply() {
        const isAr = this.currentLang === 'ar';
        document.documentElement.lang = this.currentLang;
        document.documentElement.dir = isAr ? 'rtl' : 'ltr';

        // Dispatch event for UI components to re-render
        window.dispatchEvent(new CustomEvent(EVENTS.LANG_CHANGE, {
            detail: {
                lang: this.currentLang,
                t: translations[this.currentLang]
            }
        }));
    }

    // Get current translations
    get t() {
        return translations[this.currentLang];
    }
}

export const i18n = new I18nManager();