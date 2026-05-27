# FaceID — Real-Time Face Recognition

## 🚀 How to Run in GitHub Codespaces

### Step 1 — Make sure your repo has this structure:
```
your-repo/
├── .devcontainer/
│   └── devcontainer.json      ← MUST be in this folder
├── .github/
│   └── workflows/
│       └── ci.yml
├── templates/
│   └── index.html
├── tests/
│   └── test_api.py
├── database/                  ← created automatically
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README.md
```

### Step 2 — Open Codespace
Click **Code → Codespaces → Create codespace on main**

Codespaces will automatically:
1. Pull the Python 3.11 devcontainer image
2. Run `pip install -r requirements.txt`
3. Start `python app.py`

### Step 3 — Open the app
Go to the **Ports** tab → click the link next to port **5000**

---

## 🐳 Run with Docker (locally)
```bash
docker compose up --build
# open http://localhost:5000
```

## 🐍 Run without Docker (locally)
```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```
