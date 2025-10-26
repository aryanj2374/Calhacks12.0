import React, { useState } from "react";
import "./App.css";

function App() {
  // Chat messages with sender field
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! How can I help you today?", sender: "bot" },
  ]);
  const [input, setInput] = useState("");

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

  // Handle sending a chat message
  const handleSend = () => {
    if (!input.trim()) return;

    // Add user message
    const userMessage = { id: Date.now(), text: input, sender: "user" };
    setMessages([...messages, userMessage]);
    setInput("");

    // Simulate bot response
    setTimeout(() => {
      const botMessage = {
        id: Date.now() + 1,
        text: "This is a response from the bot.",
        sender: "bot",
      };
      setMessages((prev) => [...prev, botMessage]);
    }, 500);
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
        <div className="chat">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`chat-message ${msg.sender}`}
            >
              {msg.text}
            </div>
          ))}
        </div>
        <div className="chat-input">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            onKeyPress={(e) => e.key === "Enter" && handleSend()}
          />
          <button onClick={handleSend}>Send</button>
        </div>
      </div>

      {/* Middle: Google Calendar */}
      <div className="middle">
        <h2>Calendar</h2>
        <iframe
          src="https://calendar.google.com/calendar/embed?src=YOUR_CALENDAR_ID&ctz=America/New_York"
          title="Google Calendar"
        ></iframe>
      </div>

      {/* Right: Recommendations & To-Do */}
      <div className="right">
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
