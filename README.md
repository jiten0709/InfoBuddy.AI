# InfoBuddy.AI 🤖✨

InfoBuddy.AI is a full-stack, production-ready conversational AI application featuring a real-time streaming interface and web search capabilities, powered by LangGraph and Google's Gemini model.

![InfoBuddy.AI Demo](assets/client_ss.png)

---

## 🌟 Features

### Client-Side (Next.js)

- **Real-time Streaming:** Messages stream in token-by-token for a responsive, interactive experience.
- **Persistent Conversations:** Chat history is maintained across sessions using a `conversation_id`.
- **Search Visualization:** The UI displays the agent's thought process, including search queries and results.
- **Markdown Support:** Renders formatted responses, including code blocks, lists, and links.
- **Responsive Design:** A clean, modern UI that works seamlessly on desktop and mobile.
- **Health Check:** Automatically checks server status on load.

### Server-Side (FastAPI)

- **Advanced Agent Logic:** Uses **LangGraph** to orchestrate complex workflows between the LLM and tools.
- **Tool Integration:** Equipped with **Tavily Search** to answer questions about recent events.
- **Conversation Memory:** Remembers previous parts of the conversation for contextual responses.
- **High-Performance Backend:** Built with **FastAPI** and `asyncio` for non-blocking, concurrent request handling.
- **Production-Ready:** Includes rate limiting, graceful shutdowns, CORS, and a health check endpoint.
- **Streaming API:** Uses Server-Sent Events (SSE) for efficient, unidirectional data flow to the client.

---

## 🛠️ Tech Stack

| Category       | Technology                                  |
| -------------- | ------------------------------------------- |
| **Frontend**   | Next.js, React, TypeScript, Tailwind CSS    |
| **Backend**    | Python, FastAPI, LangChain, Uvicorn         |
| **AI / LLM**   | Google Gemini, LangGraph, Tavily Search API |
| **Deployment** | Docker (optional), Gunicorn                 |

---

## 🚀 Getting Started

You can run this project using Docker.

### Prerequisites

- **Docker & Docker Compose** (for the Docker method)
- **Node.js** (v18.x or later, for the local dev method)
- **Python** (v3.10 or later, for the local dev method)
- **API Keys** for:
  - Google (for Gemini)
  - Tavily AI

### Running with Docker

**1. Set up Environment Variables:**

- In the `server/` directory, copy `.env.example` to `.env` and add your `GOOGLE_API_KEY` and `TAVILY_API_KEY`.
- The `client/` directory is pre-configured to connect to the server at `http://localhost:8000` when run via Docker Compose. No changes are needed there.

**2. Build and Run the Application:**

Open your terminal at the root of the project and run:

```bash
docker-compose up --build
```

This command will:

- Build the Docker images for both the client and server.
- Start the containers.
- Connect them on a shared Docker network.

The application will be available at:

- **Frontend:** `http://localhost:3000`
- **Backend:** `http://localhost:8000`

**Common Docker Commands:**

- **Start:** `docker-compose up -d` (runs in detached mode)
- **Stop:** `docker-compose down`
- **View Logs:** `docker-compose logs -f`

---

## 🔑 Environment Variables

### Server (`server/.env`)

```
# API Keys for AI services
GOOGLE_API_KEY="your_google_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```

### Client (`client/.env.local`)

```
# URL of the backend server
NEXT_PUBLIC_SERVER_URL="http://localhost:8000"
```

---

## 🌐 API Endpoints

The server exposes the following endpoints:

- **`GET /health`**

  - Checks the health of the server and its components.
  - **Response:**
    ```json
    {
      "status": "healthy",
      "service": "InfoBuddy.AI Server",
      "model": "gemini-2.5-flash",
      "components": { "llm": "ok", "search_tool": "ok", ... }
    }
    ```

- **`GET /chat_stream/{message}`**
  - The main endpoint for sending a message and receiving a streamed response.
  - **Query Parameters:**
    - `conversation_id` (optional): The ID of an ongoing conversation.
  - **Response:** A stream of Server-Sent Events (`text/event-stream`).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or improvements.

1.  **Fork** the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  **Commit** your changes (`git commit -m 'Add some feature'`).
5.  **Push** to the branch (`git push origin feature/YourFeature`).
6.  Open a **Pull Request**.

---

## ~ Made with ❤️ by Jiten!
