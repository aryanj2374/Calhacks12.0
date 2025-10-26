import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  const [calendarView, setCalendarView] = useState("AGENDA");

  const calendarSrc = useMemo(() => {
    const base = `https://calendar.google.com/calendar/embed`;
    const params = new URLSearchParams({
      src: calendarId,
      ctz: calendarTz,
      mode: calendarView,
      showCalendars: "0",
      showTabs: "0",
      showTitle: "0",
    });
    return `${base}?${params.toString()}`;
  }, [calendarId, calendarTz, calendarView]);

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
  const [showIdeas, setShowIdeas] = useState(true);
  const [pendingConfirmation, setPendingConfirmation] = useState(null);
  const chatRef = useRef(null);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/status`);
      if (!response.ok) {
        throw new Error("Status request failed");
      }
      const payload = await response.json();
      setSyncInfo(payload.gmail_sync || null);
    } catch (err) {
      console.error("Failed to fetch status", err);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 60000);
    return () => clearInterval(id);
  }, [fetchStatus]);

  const formatSyncTime = (timestamp, options = {}) => {
    if (!timestamp) return "—";
    try {
      return new Date(timestamp).toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
        ...options,
      });
    } catch {
      return timestamp;
    }
  };

  const syncErrors = syncInfo?.errors || [];

  const [recommendations] = useState([
    "Skim tomorrow's lecture slides",
    "Confirm study group time",
    "Block focus time for project",
  ]);

  const [todos, setTodos] = useState([
    { id: 1, text: "Final project milestone", removing: false },
    { id: 2, text: "Send mentor update", removing: false },
    { id: 3, text: "Revisit HW feedback", removing: false },
  ]);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || sending) return;
    const userMessage = { id: Date.now(), text: input.trim(), sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setStatus("Thinking...");
    setError("");
    setSending(true);

    try {
      let response, payload;
      
      // Check if we're handling a confirmation response
      if (pendingConfirmation) {
        response = await fetch(`${apiBaseUrl}/api/confirm`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            response: userMessage.text,
            pending_event: pendingConfirmation
          }),
        });
        setPendingConfirmation(null);
      } else {
        response = await fetch(`${apiBaseUrl}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: userMessage.text }),
        });
      }
      
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
        needsConfirmation: payload.needs_confirmation,
      };
      setMessages((prev) => [...prev, botMessage]);
      
      // Handle confirmation requests
      if (payload.needs_confirmation && payload.pending_event) {
        setPendingConfirmation(payload.pending_event);
        setStatus("Waiting for your confirmation...");
      } else {
        setPendingConfirmation(null);
        if (payload.action === "import_course") {
          setStatus(
            payload.executed
              ? "Course imported and synced to Google Calendar."
              : "Course imported (dry run)."
          );
        } else {
          setStatus(payload.executed ? "Calendar updated." : "Preview only (dry run).");
        }
      }
      setShowIdeas(false);
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
      setPendingConfirmation(null);
    } finally {
      setSending(false);
    }
  };

  const handleRemove = (id) => {
    setTodos((prev) =>
      prev.map((todo) =>
        todo.id === id ? { ...todo, removing: true } : todo
      )
    );
    setTimeout(() => {
      setTodos((prev) => prev.filter((todo) => todo.id !== id));
    }, 260);
  };

  const lastCalendarUpdate = syncInfo?.applied
    ? formatSyncTime(syncInfo.timestamp)
    : "Awaiting live sync";

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="pill">Agentic Workspace</p>
          <h1>Your calendar co-pilot</h1>
          <p className="subtitle">
            Capture intents, scrape syllabi, and stay aligned with your courses without leaving chat.
          </p>
          <div className="hero-metrics">
            <div className="metric">
              <span className="metric-label">Status</span>
              <span className="metric-value">{sending ? "Thinking…" : "Ready"}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Last calendar update</span>
              <span className="metric-value">{lastCalendarUpdate}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Gmail sync</span>
              <span className={`metric-pill ${syncInfo?.applied ? "ok" : "warn"}`}>
                {syncInfo
                  ? syncInfo.applied
                    ? "UPDATED"
                    : "PREVIEW"
                  : "WAITING"}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="dashboard-layout">
        <div className="top-grid">
          <section className="panel calendar-panel">
            <div className="panel-header">
              <h2>Google Calendar</h2>
              <span className="panel-subtitle">Live embed of your connected calendar</span>
            </div>
            <div className="calendar-controls">
              {[
                { label: "Week", mode: "WEEK" },
                { label: "Month", mode: "MONTH" },
                { label: "Schedule", mode: "AGENDA" },
              ].map(({ label, mode }) => (
                <button
                  key={mode}
                  type="button"
                  className={calendarView === mode ? "active" : ""}
                  onClick={() => setCalendarView(mode)}
                  aria-pressed={calendarView === mode}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="calendar-frame">
              <iframe src={calendarSrc} title="Google Calendar"></iframe>
            </div>
          </section>

          <section className="panel chat-panel">
            <div className="panel-header">
              <h2>Chat</h2>
              <span className="panel-subtitle">Natural language control center</span>
            </div>
          <div className="chat-window" ref={chatRef}>
            {messages.map((msg) => (
                <div key={msg.id} className={`chat-bubble ${msg.sender} ${msg.needsConfirmation ? 'confirmation-request' : ''}`}>
                  <div className="bubble-top">
                    <span className="bubble-role">{msg.sender === "bot" ? "Agent" : "You"}</span>
                    {msg.action && (
                      <span className="bubble-meta">
                        {msg.action} · {msg.executed ? "executed" : "dry run"}
                      </span>
                    )}
                    {msg.needsConfirmation && (
                      <span className="bubble-meta confirmation-badge">⚠️ CONFLICT DETECTED</span>
                    )}
                  </div>
                  <p>{msg.text}</p>
                  {msg.error && <span className="bubble-error">error</span>}
              </div>
            ))}
          </div>
            {showIdeas && (
              <div className="ideas-card">
                <p className="ideas-label">Need ideas?</p>
                <p className="ideas-text">
                  “Add office hours tomorrow at 4”, “Import CS61A syllabus”, “Move HW4 to Friday 9pm”.
                </p>
              </div>
            )}
            <div className="composer">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={pendingConfirmation ? "Type 'yes' to confirm or 'no' to cancel…" : "Ask to add, move, or import events…"}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                disabled={sending}
              />
            <button className="send-button" onClick={handleSend} disabled={sending}>
              {sending ? "…" : <span aria-label="Send" role="img">➤</span>}
            </button>
          </div>
            <div className="status-inline">
              <span>{status}</span>
              {error && <span className="status-error">{error}</span>}
            </div>
          </section>
        </div>

        <section className="panel insights-panel overview-panel">
          <div className="card sync-card">
            <div className="sync-card-header">
              <div>
                <p className="sync-label">Last Gmail sync</p>
                <p className="sync-time">{formatSyncTime(syncInfo?.timestamp)}</p>
                <p className="sync-note subtle">
                  Calendar last updated: {lastCalendarUpdate}
                </p>
              </div>
              <span className={`sync-pill ${syncInfo?.applied ? "success" : "pending"}`}>
                {syncInfo
                  ? syncInfo.applied
                    ? "UPDATED"
                    : "PREVIEW"
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

          <div className="card rec-card">
            <h3>Recommendations</h3>
            <ul className="pill-list">
              {recommendations.map((rec, idx) => (
                <li key={idx}>{rec}</li>
              ))}
            </ul>
          </div>
          <div className="card todo-card">
            <h3>To-do</h3>
            <ul className="todo-list">
              {todos.map((todo) => (
                <li key={todo.id} className={todo.removing ? "removing" : ""}>
                  <label>
                    <input type="checkbox" onChange={() => handleRemove(todo.id)} />
                    <span>{todo.text}</span>
                  </label>
                </li>
              ))}
            </ul>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
