# Health AI Chat — Test Client

A minimal React Native (Expo) chatbot that connects to your Health AI FastAPI backend.

---

## Prerequisites

- Node.js 18+ on your laptop  
- Expo Go app on your phone ([iOS](https://apps.apple.com/app/expo-go/id982107779) / [Android](https://play.google.com/store/apps/details?id=host.exp.exponent))  
- Both laptop and phone on the **same WiFi network**

---

## Step 1 — Start the AI server on your laptop

```bash
cd path/to/aichatbot_v2_fixed
python -m health_ai.api.main
```

Look for this line in the output:

```
  Network: http://192.168.x.x:8000   ← use this on the phone
```

Write down that IP address.

---

## Step 2 — Install and run the React Native app

```bash
cd HealthAIChat
npm install
npx expo start
```

Expo will show a QR code in the terminal.

---

## Step 3 — Open on your phone

1. Open the **Expo Go** app on your phone  
2. Scan the QR code shown in the terminal  
3. The app will load

---

## Step 4 — Connect to your server

1. The app opens with a ⚙️ settings dialog  
2. Enter your **laptop's LAN IP** (e.g. `192.168.1.42`)  
3. Leave port as `8000` (or change if you used `--port`)  
4. Tap **Test Connection** — you should see ✓ Server reachable!  
5. Tap **Save**

---

## Step 5 — Chat!

Type a question or tap one of the suggestion chips:

- "What are my abnormal lab results?"  
- "What medicines am I prescribed?"  
- "Summarise all my test results"  

The app sends your message to the server → the server runs RAG + LLM → reply appears in chat.

---

## API endpoint used

```
POST http://<server-ip>:8000/query/profile-1
Body: { "query": "...", "history": ["prev Q", "prev A", ...] }
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Could not reach the server" | Make sure laptop and phone are on the same WiFi |
| App loads but shows error | Check the server is running (`python -m health_ai.api.main`) |
| QR code won't scan | Try pressing `w` in the Expo terminal to open in browser first |
| Port conflict | Start server with `--port 9000` and update the port in the app |

---

## Project structure

```
HealthAIChat/
  App.js          ← entire app in one file (chat UI + config modal)
  package.json
  app.json
  babel.config.js
  assets/
```
