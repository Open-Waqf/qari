// app/src/core/i18n.ts
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

        // --- Onboarding Modal ---
        welcomeTitle: "Welcome to Qari Finder",
        welcomeDesc: "Identify reciters instantly and privately.",
        getStarted: "Get Started",
        calibTitle: "Calibrating Ears",
        calibDesc: "We are adjusting to your room's noise levels. Please stay silent...",
        scienceTitle: "Science of Sound",
        scienceDesc: "Local neural networks analyze frequency patterns without recording audio.",
        back: "Back"
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

        // --- Onboarding Modal (Translated) ---
        welcomeTitle: "مرحباً بك في قارئ فايندر",
        welcomeDesc: "تعرف على القراء فورياً وبخصوصية تامة.",
        getStarted: "ابدأ الآن",
        calibTitle: "ضبط مستوى السمع",
        calibDesc: "نقوم بضبط حساسية الميكروفون حسب ضجيج الغرفة. يرجى التزام الصمت...",
        scienceTitle: "كيف يعمل النظام؟",
        scienceDesc: "تقوم الشبكات العصبية المحلية بتحليل أنماط الترددات دون الحاجة لتسجيل الصوت.",
        back: "رجوع"
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
        document.documentElement.lang = this.currentLang;
        document.documentElement.dir = isAr ? 'rtl' : 'ltr';

        window.dispatchEvent(new CustomEvent('lang-change', {
            detail: {lang: this.currentLang, t: translations[this.currentLang]}
        }));
    }

    get t() {
        return translations[this.currentLang];
    }
}

export const i18n = new I18nManager();