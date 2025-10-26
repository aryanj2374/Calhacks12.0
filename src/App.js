import React, { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";

function App() {
  const apiBaseUrl = useMemo(
    () => process.env.REACT_APP_API_BASE_URL || "http://localhost:8000",
    []
  );
  const calendarId = useMemo(
    () => process.env.REACT_APP_GOOGLE_CALENDAR_ID || "primary",
    []
  );
  const calendarTz = useMemo(
    () => process.env.REACT_APP_GOOGLE_CALENDAR_TZ || "America/Los_Angeles",
    []
  );
  const calendarSrc = useMemo(
    () =>
      `https://calendar.google.com/calendar/embed?src=${encodeURIComponent(
        calendarId
      )}&ctz=${encodeURIComponent(calendarTz)}`,
    [calendarId, calendarTz]
  );
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hi! I'm your calendar agent. Ask me to add or move events, or paste a course syllabus link and mention 'course' to import it.",
      sender: "bot",
    },
  ]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("Connected to local agent.");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const [syncInfo, setSyncInfo] = useState(null);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/status`);
      if (!response.ok) {
        throw new Error("Status request failed");
      }
      const payload = await response.json();
      setSyncInfo(payload.gmail_sync || null);
    } catch (err) {
      // Keep existing status but surface console diagnostic
      console.error("Failed to fetch status", err);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 60000);
    return () => clearInterval(id);
  }, [fetchStatus]);

  const formatSyncTime = (timestamp) => {
    if (!timestamp) return "—";
    try {
      return new Date(timestamp).toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      });
    } catch {
      return timestamp;
    }
  };
  const syncErrors = syncInfo?.errors || [];

  // Example Recommendations
  const [recommendations] = useState([
    "Read Chapter 3",
    "Finish coding exercise",
    "Prepare for quiz",
  ]);

  // To-Do list with unique IDs
  const [todos, setTodos] = useState([
    { id: 1, text: "Finish homework" },
    { id: 2, text: "Call Alice" },
    { id: 3, text: "Read book" },
  ]);

  const handleSend = async () => {
    if (!input.trim() || sending) return;

    const userMessage = { id: Date.now(), text: input.trim(), sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setStatus("Thinking...");
    setError("");
    setSending(true);

    try {
      const response = await fetch(`${apiBaseUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage.text }),
      });

      let payload = null;
      if (response.ok) {
        payload = await response.json();
      } else {
        const detail = await response.json().catch(() => ({}));
        const message = detail?.detail || "Agent request failed.";
        throw new Error(message);
      }

      const botMessage = {
        id: Date.now() + 1,
        text: payload.reply,
        sender: "bot",
        action: payload.action,
        executed: payload.executed,
      };
      setMessages((prev) => [...prev, botMessage]);

      if (payload.action === "import_course") {
        setStatus(
          payload.executed
            ? "Course imported and synced to Google Calendar."
            : "Course imported (dry run)."
        );
      } else {
        setStatus(payload.executed ? "Calendar updated." : "Preview only (dry run).");
      }
    } catch (err) {
      const botMessage = {
        id: Date.now() + 2,
        text: err.message || "Unexpected error talking to the agent.",
        sender: "bot",
        error: true,
      };
      setMessages((prev) => [...prev, botMessage]);
      setStatus("Unable to complete the request.");
      setError(err.message || "Request failed.");
    } finally {
      setSending(false);
    }
  };

  // Handle To-Do removal with smooth animation
  const handleRemove = (id) => {
    setTodos((prev) =>
      prev.map((todo) =>
        todo.id === id ? { ...todo, removing: true } : todo
      )
    );
    setTimeout(() => {
      setTodos((prev) => prev.filter((todo) => todo.id !== id));
    }, 300); // match CSS transition
  };

  return (
    <div className="container">
      {/* Left: Chatbot */}
      <div className="left">
        <h2>Chatbot</h2>
        <p className="chat-hint">
          Ask for new meetings, reschedules, or paste your course link with a note like
          “import this course” to trigger the scraper.
        </p>
        <div className="chat">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`chat-message ${msg.sender}`}
            >
              <span>{msg.text}</span>
              {msg.action && (
                <span className="chat-meta">
                  {msg.action} · {msg.executed ? "executed" : "dry run"}
                </span>
              )}
              {msg.error && <span className="chat-meta error">Error</span>}
            </div>
          ))}
        </div>
        <div className="status-bar">
          {sending ? "Sending..." : status}
          {error && <span className="status-error">{error}</span>}
        </div>
        <div className="chat-input">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            disabled={sending}
          />
          <button onClick={handleSend} disabled={sending}>
            {sending ? "…" : "Send"}
          </button>
        </div>
      </div>

      {/* Middle: Google Calendar */}
      <div className="middle">
        <h2>Calendar</h2>
        <iframe src={calendarSrc} title="Google Calendar"></iframe>
      </div>

      {/* Right: Recommendations & To-Do */}
      <div className="right">
        <div className="sync-card">
          <div className="sync-card-header">
            <div>
              <p className="sync-label">Last Gmail Sync</p>
              <p className="sync-time">{formatSyncTime(syncInfo?.timestamp)}</p>
            </div>
            <span
              className={`sync-pill ${
                syncInfo?.applied ? "success" : "pending"
              }`}
            >
              {syncInfo
                ? syncInfo.applied
                  ? "CALENDAR UPDATED"
                  : "PREVIEW ONLY"
                : "WAITING"}
            </span>
          </div>
          {syncInfo ? (
            <>
              <div className="sync-meta">
                <span>{syncInfo.event_count} events parsed</span>
                <span>{syncInfo.created_count} added</span>
              </div>
              {syncErrors.length ? (
                <ul className="sync-errors">
                  {syncErrors.slice(0, 2).map((msg, idx) => (
                    <li key={idx}>{msg}</li>
                  ))}
                  {syncErrors.length > 2 && (
                    <li>+{syncErrors.length - 2} more</li>
                  )}
                </ul>
              ) : (
                <p className="sync-note">Inbox looks good!</p>
              )}
            </>
          ) : (
            <p className="sync-note">Waiting for the background sync…</p>
          )}
        </div>

        {/* Recommendations */}
        <div className="recommendations">
          <h2>Recommendations</h2>
          <ul>
            {recommendations.map((rec, idx) => (
              <li key={idx}>{rec}</li>
            ))}
          </ul>
        </div>

        {/* To-Do List */}
        <div className="todo">
          <h2>To-Do List</h2>
          <ul>
            {todos.map((todo) => (
              <li
                key={todo.id}
                className={`todo-item ${todo.removing ? "removing" : ""}`}
              >
                <label>
                  <input
                    type="checkbox"
                    onChange={() => handleRemove(todo.id)}
                  />
                  <span>{todo.text}</span>
                </label>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default App;
