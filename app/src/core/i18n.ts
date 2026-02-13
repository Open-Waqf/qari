export type Lang = 'en' | 'ar';

export const translations = {
    en: {
        appTitle: "Qari",
        appAccent: "Finder",
        subtitle: "Neural Voice Identification",
        acousticLabel: "Acoustic Signature",
        awaiting: "Awaiting Signal",
        imitationTitle: "Imitation Analysis",
        install: "📥 Install App",
        initBtn: "Initialize Engine",
        loading: "Loading Brain...",
        retry: "Retry Connection",
        online: "🚀 System Online",
        analyzing: "Listening...",
        stabilizing: "Verifying Match...",
        tooQuiet: "Too Quiet - Move Closer",
        confirmed: "High Confidence",
        unknown: "Unsure - Keep Listening",
    },
    ar: {
        appTitle: "قارئ",
        appAccent: "فايندر",
        subtitle: "التعرف الذكي على القراء",
        acousticLabel: "البصمة الصوتية",
        awaiting: "بانتظار الإشارة",
        imitationTitle: "تحليل المحاكاة",
        install: "📥 تثبيت التطبيق",
        initBtn: "تشغيل المحرك",
        loading: "جاري تحميل النموذج...",
        retry: "إعادة المحاولة",
        online: "🚀 النظام جاهز",
        analyzing: "جاري الاستماع...",
        stabilizing: "التحقق من التطابق...",
        tooQuiet: "الصوت منخفض - اقترب",
        confirmed: "تطابق قوي",
        unknown: "غير مؤكد - استمر في القراءة",
    }
};

class I18nManager {
    currentLang: Lang = 'en';

    toggle(): Lang {
        this.currentLang = this.currentLang === 'en' ? 'ar' : 'en';
        this.apply();
        return this.currentLang;
    }

    apply() {
        const isAr = this.currentLang === 'ar';

        // 1. Set Direction
        document.documentElement.lang = this.currentLang;
        document.documentElement.dir = isAr ? 'rtl' : 'ltr';

        // 2. Dispatch Event for Components
        window.dispatchEvent(new CustomEvent('lang-change', {
            detail: {lang: this.currentLang, t: translations[this.currentLang]}
        }));
    }

    get t() {
        return translations[this.currentLang];
    }
}

export const i18n = new I18nManager();