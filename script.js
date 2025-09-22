document.addEventListener("DOMContentLoaded", () => {
    const menuToggleButton = document.getElementById("menu-toggle-button");
    const sidebar = document.querySelector(".sidebar");
    const closeSidebarButton = document.getElementById("close-sidebar");
    const sendButton = document.getElementById("sendButton");
    const userInput = document.getElementById("userInput");
    const conversation = document.getElementById("conversation");
    const newChatBtn = document.querySelector(".new-chat-btn");
    const historyList = document.getElementById("historyList");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    const chatContainer = document.querySelector(".chat-container");

    // ===== Config =====
    const GREETING_DELAY_MS = 600;
    const GREETING_PATTERN = /^\s*(hi|hello|hey)\b/i;
    const GREETING_REPLY = {
        title: "👋 Hello!",
        answer: "I am <strong>H.A.N.D.Y AI</strong> 🤖. I’ll answer your questions based on the student handbook.",
        sources: [],
        notes: "Ask me about any policy, and I’ll help you find it."
    };

    let isAnswering = false;
    let controller = null;
    let history = JSON.parse(localStorage.getItem("chatHistory") || "[]");

    // ===== Sidebar =====
    menuToggleButton.addEventListener("click", () => {
        sidebar.classList.add("active");
        chatContainer.classList.add("shifted");
    });

    closeSidebarButton.addEventListener("click", () => {
        sidebar.classList.remove("active");
        chatContainer.classList.remove("shifted");
    });

    // 🚫 Removed outside click listener
    // (sidebar closes ONLY with the X button now)

    // ===== Chat Helpers =====
    function getTimeString() {
        return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }

    function addMessage(content, sender, isStructured = false, isGreeting = false) {
        const wrapper = document.createElement("div");
        wrapper.classList.add("msg", sender);

        if (sender === "bot") {
            if (isGreeting) {
                wrapper.innerHTML = `
                    <div class="bot-meta">🤖 H.A.N.D.Y AI • Today at ${getTimeString()}</div>
                    <div class="bot-text greeting">${content.answer}</div>
                `;
            } else if (isStructured && typeof content === "object") {
                let sourcesHTML = "";
                if (content.sources && content.sources.length > 0) {
                    sourcesHTML = `
                        <h4>Sources:</h4>
                        <div class="bot-section sources">
                            <ul>
                                ${content.sources.map(src => `<li>${src}</li>`).join("")}
                            </ul>
                        </div>`;
                }
                let notesHTML = content.notes
                    ? `<div class="bot-section"><strong>Note:</strong> ${content.notes}</div>`
                    : "";

                wrapper.innerHTML = `
                    <div class="bot-meta">🤖 H.A.N.D.Y AI • Today at ${getTimeString()}</div>
                    <div class="bot-text">
                        <h3>${content.title || "Answer"}</h3>
                        <div>${content.answer || ""}</div>
                        ${sourcesHTML}
                        ${notesHTML}
                    </div>`;
            } else {
                wrapper.innerHTML = `
                    <div class="bot-meta">🤖 H.A.N.D.Y AI • Today at ${getTimeString()}</div>
                    <div class="bot-text">${marked.parse(content)}</div>`;
            }
        } else {
            const bubble = document.createElement("div");
            bubble.classList.add("user-bubble");
            bubble.textContent = content;
            wrapper.appendChild(bubble);
        }

        conversation.appendChild(wrapper);
        conversation.scrollTop = conversation.scrollHeight;
        return wrapper;
    }

    function addSearching(query) {
        const wrapper = document.createElement("div");
        wrapper.classList.add("msg", "bot");
        wrapper.innerHTML = `
            <div class="bot-meta">🤖 H.A.N.D.Y AI • Today at ${getTimeString()}</div>
            <div class="bot-text">
                <span class="searching-text">Searching for "${query}"</span>
            </div>`;
        conversation.appendChild(wrapper);
        conversation.scrollTop = conversation.scrollHeight;
        return wrapper;
    }

    // ===== Input Locking =====
    function lockInput() {
        isAnswering = true;
        userInput.disabled = true;
        sendButton.innerHTML = "Cancel";
        sendButton.classList.add("cancel");
    }

    function unlockInput() {
        isAnswering = false;
        userInput.disabled = false;
        sendButton.innerHTML = `<i class="fas fa-paper-plane"></i>`;
        sendButton.classList.remove("cancel");
    }

    function cancelRequest() {
        if (controller) controller.abort();
        addMessage("❌ Request canceled.", "bot");
        unlockInput();
    }

    // ===== History =====
    function updateHistoryUI() {
        historyList.innerHTML = "";
        history.forEach((item) => {
            const li = document.createElement("li");
            li.textContent = item.query;
            li.title = item.query;
            li.addEventListener("click", () => {
                conversation.innerHTML = "";
                addMessage(item.query, "user");
                if (item.structured) {
                    addMessage(item.answer, "bot", true);
                } else {
                    addMessage(item.answer, "bot");
                }
                sidebar.classList.remove("active");
                chatContainer.classList.remove("shifted");
            });
            historyList.appendChild(li);
        });
        localStorage.setItem("chatHistory", JSON.stringify(history));
    }

    updateHistoryUI();

    clearHistoryBtn.addEventListener("click", () => {
        if (confirm("Clear all history?")) {
            history = [];
            localStorage.removeItem("chatHistory");
            updateHistoryUI();
            sidebar.classList.remove("active");
            chatContainer.classList.remove("shifted");
        }
    });

    // ===== Send Message =====
    function sendMessage() {
        if (isAnswering) {
            cancelRequest();
            return;
        }

        const query = userInput.value.trim();
        if (!query) return;

        addMessage(query, "user");
        userInput.value = "";

        lockInput();
        controller = new AbortController();
        const signal = controller.signal;

        if (GREETING_PATTERN.test(query)) {
            const thinking = addSearching(query);
            setTimeout(() => {
                if (thinking && thinking.parentNode) conversation.removeChild(thinking);
                addMessage(GREETING_REPLY, "bot", false, true);
                unlockInput();
                history.push({ query, answer: GREETING_REPLY, structured: true });
                updateHistoryUI();
            }, GREETING_DELAY_MS);
            return;
        }

        const searchingMsg = addSearching(query);
        fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query }),
            signal
        })
        .then(res => res.json())
        .then(data => {
            if (searchingMsg && searchingMsg.parentNode) conversation.removeChild(searchingMsg);

            if (data.title || data.sources || data.notes) {
                addMessage(data, "bot", true);
                history.push({ query, answer: data, structured: true });
            } else {
                addMessage(data.answer, "bot");
                history.push({ query, answer: data.answer, structured: false });
            }

            updateHistoryUI();
        })
        .catch(err => {
            if (searchingMsg && searchingMsg.parentNode) conversation.removeChild(searchingMsg);
            if (err.name === "AbortError") {
                addMessage("❌ Request aborted.", "bot");
            } else {
                addMessage("⚠ Error: Could not reach server.", "bot");
            }
        })
        .finally(() => {
            unlockInput();
        });
    }

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });

    newChatBtn.addEventListener("click", () => {
        conversation.innerHTML = "";
    });
});
