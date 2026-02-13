## Contributing to Qari Finder 🌙

First off, thank you for considering contributing! Qari Finder is a community-driven project aimed at preserving and identifying the beautiful voices of Quran reciters through technology.

## 🛠️ Development Setup

The project is split into two main environments:

1.  **The Web App (`/app`):** Built with Lit, Vite, and TypeScript.
2.  **The Research Lab (`/research`):** A Python-based suite for training the AI models.

### Local Environment
- **Frontend:** Node.js (v18+) and npm.
- **AI Research:** Python 3.9+ (Virtual environment highly recommended).

---

## 🎤 How to Add a New Reciter (Qari)

Adding a new voice is the most valuable way to contribute. Follow these steps:

1.  **Data Collection:**
    * Create a folder in `research/audio/<reciter_name>/`.
    * Add high-quality MP3 clips (ideally 3-5 minutes each) of the reciter.
2.  **Preprocessing:**
    * Run `python prepare_data.py` to convert audio into spectrograms (`.npz` files).
3.  **Training:**
    * Run `python train.py`. This will train the Keras model (`qari_model.h5`).
4.  **Web Conversion:**
    * Run `python convert_wizard.py`. This exports the model to the `app/public/models/tfjs_model/` directory so the web app can use it.

---

## 💻 Code Contribution Guidelines

### Branching Policy
- `main`: Production-ready code.
- `develop`: Ongoing feature work.
- Please create a feature branch (`feature/your-feature-name`) before submitting a Pull Request.

### Code Style
- **TypeScript:** We use standard ESLint and Prettier configurations. Please run `npm run typecheck` in the `/app` folder before committing.
- **Python:** Follow PEP 8 guidelines.

### Submitting a Pull Request
1.  Fork the repo and create your branch from `develop`.
2.  If you added a new model, include a brief report on the new accuracy metrics.
3.  Ensure all tests pass (`npm run test:unit` in `/app`).
4.  Open a PR with a clear description of your changes.

---

## ⚖️ License
By contributing, you agree that your contributions will be licensed under the **GPL-3.0-or-later** license.