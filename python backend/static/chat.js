document.addEventListener("DOMContentLoaded", () => {
  const layout = document.querySelector(".chat-layout");
  const chatBox = document.getElementById("chat");
  const input = document.getElementById("msg");
  const sendBtn = document.getElementById("send-btn");
  const keywordsEl = document.getElementById("keywords");

  if (!layout || !chatBox || !input || !sendBtn) {
    return;
  }

  const room = layout.dataset.room;
  const username = layout.dataset.username || "you";

  const scrollToBottom = () => {
    chatBox.scrollTo({
      top: chatBox.scrollHeight,
      behavior: "smooth",
    });
  };

  const createMessageElement = (sender, text, isBot, animate = true) => {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${isBot ? "bot" : "user"}${
      animate ? " message-animate-in" : ""
    }`;

    const meta = document.createElement("div");
    meta.className = "message-meta";
    const senderSpan = document.createElement("span");
    senderSpan.className = "message-sender";
    senderSpan.textContent = isBot ? "Academix AI" : sender;
    meta.appendChild(senderSpan);

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    if (typeof marked !== "undefined") {
      bubble.innerHTML = marked.parse(text);
    } else {
      bubble.textContent = text;
    }

    wrapper.appendChild(meta);
    wrapper.appendChild(bubble);

    return wrapper;
  };

  const setLoadingState = (isLoading) => {
    if (isLoading) {
      sendBtn.classList.add("is-loading");
      sendBtn.disabled = true;
    } else {
      sendBtn.classList.remove("is-loading");
      sendBtn.disabled = false;
    }
  };

  const updateKeywords = (keywords) => {
    if (!keywordsEl) return;
    keywordsEl.innerHTML = "";

    if (!keywords || !keywords.length) {
      const p = document.createElement("p");
      p.className = "keyword-empty";
      p.textContent = "Keywords will appear here as you chat.";
      keywordsEl.appendChild(p);
      return;
    }

    keywords.forEach((k) => {
      const chip = document.createElement("span");
      chip.className = "keyword-chip keyword-animate-in";
      chip.textContent = k;
      keywordsEl.appendChild(chip);
    });
  };

  const send = async () => {
    const text = input.value.trim();
    if (!text) return;

    // Optimistically render user message
    const userMsgEl = createMessageElement(username, text, false, true);
    chatBox.appendChild(userMsgEl);
    scrollToBottom();

    input.value = "";
    setLoadingState(true);

    try {
      const res = await fetch(`/send/${encodeURIComponent(room)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const botMsgEl = createMessageElement("Academix AI", "", true, true);
      chatBox.appendChild(botMsgEl);
      scrollToBottom();
      const bubble = botMsgEl.querySelector('.message-bubble');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let rawText = "";
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        
        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const message = buffer.slice(0, boundary).trim();
          buffer = buffer.slice(boundary + 2);
          
          if (message.startsWith("data: ")) {
            try {
              const data = JSON.parse(message.substring(6));
              if (data.chunk) {
                rawText += data.chunk;
                if (typeof marked !== "undefined") {
                  bubble.innerHTML = marked.parse(rawText);
                } else {
                  bubble.textContent = rawText;
                }
                scrollToBottom();
              } else if (data.keywords) {
                updateKeywords(data.keywords);
              } else if (data.error) {
                console.error("Stream error:", data.error);
                rawText += "\n\n**Error:** " + data.error;
                if (typeof marked !== "undefined") {
                  bubble.innerHTML = marked.parse(rawText);
                } else {
                  bubble.textContent = rawText;
                }
              }
            } catch (e) {
              console.error("Parse error", e);
            }
          }
          boundary = buffer.indexOf("\n\n");
        }
      }
    } catch (e) {
      const errorMsg = createMessageElement(
        "System",
        "Something went wrong sending your message. Please try again.",
        true,
        true
      );
      chatBox.appendChild(errorMsg);
      scrollToBottom();
    } finally {
      setLoadingState(false);
      input.focus();
    }
  };

  sendBtn.addEventListener("click", send);

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  // Parse existing static messages
  const staticBubbles = document.querySelectorAll(".message-static .message-bubble");
  staticBubbles.forEach(bubble => {
    if (typeof marked !== "undefined") {
      bubble.innerHTML = marked.parse(bubble.textContent.trim());
    }
  });

  // Initial scroll
  scrollToBottom();
});

async function sendMessage() {
    const input = document.getElementById("userInput");
    const chatBox = document.getElementById("chatBox");

    const text = input.value.trim();
    if (!text) return;

    addMessage(text, "user");
    input.value = "";

    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    });

    const data = await response.json();
    addMessage(data.reply, "bot");
}

function addMessage(text, sender) {
    const chatBox = document.getElementById("chatBox");
    const msg = document.createElement("div");

    msg.classList.add("message", sender);
    msg.innerText = text;

    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}
