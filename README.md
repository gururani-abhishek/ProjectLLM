# SDLC RAG Chatbot

This project provides a Retrieval-Augmented Generation (RAG) chatbot for SDLC (Software Development Lifecycle) queries, powered by FastAPI and OpenAI.

---

## Setup Instructions

### 1. Generate and Store OpenAI API Key

- Get your OpenAI API key from [OpenAI](https://platform.openai.com/).
- Create a `.env` file in your project root with the following content:
  ```
  OPENAI_API_KEY=your_openai_api_key_here
  ```
- The code will automatically load your API key from the `.env` file.

---

### 2. Start the Virtual Environment

```bash
source .venv/bin/activate
```

---

### 3. Start the FastAPI Backend

```bash
uvicorn app.api_chatbot:app --reload
```

---

### 4. Serve the Chat Frontend

Open a **new terminal window**, navigate to the `static` directory, and run:

```bash
python3 -m http.server 8080
```

_(Make sure you're inside the `static` directory where `chat.html` is located.)_

---

### 5. Open the Chatbot Page

In your browser, go to:

[http://localhost:8080/chat.html](http://localhost:8080/chat.html)

---

**Note:**

- Make sure both the FastAPI backend and the static server are running.
- For best results, always mention the SDLC or the Field name explicitly in your chat queries.
