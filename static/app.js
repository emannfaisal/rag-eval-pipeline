let registeredBooks = [];

document.addEventListener("DOMContentLoaded", async () => {
    await fetchBooks();
    setupTextareaAutoResize();
});

async function fetchBooks() {
    try {
        const res = await fetch("/api/books");
        if (res.ok) {
            registeredBooks = await res.json();
            updateBookSelectorUI();
        }
    } catch (err) {
        console.error("Failed to load books list:", err);
    }
}

function updateBookSelectorUI() {
    const select = document.getElementById("bookSelect");
    if (!select || registeredBooks.length === 0) return;

    select.innerHTML = "";
    registeredBooks.forEach(b => {
        const opt = document.createElement("option");
        opt.value = b.id;
        opt.textContent = `${b.icon} ${b.title}`;
        select.appendChild(opt);
    });

    onBookChange();
}

function onBookChange() {
    const select = document.getElementById("bookSelect");
    const activeLabel = document.getElementById("activeBookLabel");
    if (!select || !activeLabel) return;

    const selectedId = select.value;
    const book = registeredBooks.find(b => b.id === selectedId);
    if (book) {
        activeLabel.textContent = `${book.icon} ${book.title}`;
    } else {
        activeLabel.textContent = select.options[select.selectedIndex]?.text || "Selected Book";
    }
}

function sendQuickPrompt(promptText) {
    const textarea = document.getElementById("userInput");
    textarea.value = promptText;
    textarea.focus();
    handleFormSubmit(new Event("submit"));
}

function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        document.getElementById("chatForm").requestSubmit();
    }
}

function setupTextareaAutoResize() {
    const textarea = document.getElementById("userInput");
    if (!textarea) return;

    textarea.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
        if (this.scrollHeight > 128) {
            this.style.overflowY = "auto";
        } else {
            this.style.overflowY = "hidden";
        }
    });
}

async function handleFormSubmit(event) {
    if (event) event.preventDefault();

    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const bookSelect = document.getElementById("bookSelect");
    const welcomeScreen = document.getElementById("welcomeScreen");

    const question = userInput.value.trim();
    if (!question) return;

    const selectedBookId = bookSelect.value;
    const bookObj = registeredBooks.find(b => b.id === selectedBookId) || {
        title: bookSelect.options[bookSelect.selectedIndex]?.text || "Book",
        icon: "📖"
    };

    // Hide welcome screen on first message
    if (welcomeScreen) welcomeScreen.style.display = "none";

    // Append User Message
    appendUserMessage(question);

    // Reset input
    userInput.value = "";
    userInput.style.height = "auto";
    sendBtn.disabled = true;

    // Show Typing Indicator
    const typingId = showTypingIndicator(bookObj.icon);

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: question,
                book_id: selectedBookId,
                n_results: 5,
                use_reranker: true
            })
        });

        removeTypingIndicator(typingId);

        if (!response.ok) {
            const errData = await response.json();
            appendErrorMessage(errData.detail || "Error generating answer.");
            return;
        }

        const data = await response.json();
        appendAssistantMessage(data.answer, data.sources, data.book_title || bookObj.title, data.book_icon || bookObj.icon);

    } catch (err) {
        removeTypingIndicator(typingId);
        appendErrorMessage("Failed to connect to server. Ensure server.py is running!");
    } finally {
        sendBtn.disabled = false;
        scrollToBottom();
    }
}

function appendUserMessage(text) {
    const list = document.getElementById("messagesList");
    const messageEl = document.createElement("div");
    messageEl.className = "flex justify-end animate-fade-in";
    messageEl.innerHTML = `
        <div class="max-w-[85%] sm:max-w-[75%] rounded-2xl rounded-tr-sm bg-zinc-950 text-white px-4 py-3 text-sm font-medium shadow-md">
            <p class="whitespace-pre-wrap">${escapeHtml(text)}</p>
        </div>
    `;
    list.appendChild(messageEl);
    scrollToBottom();
}

function appendAssistantMessage(answerText, sources, bookTitle, bookIcon) {
    const list = document.getElementById("messagesList");
    const messageEl = document.createElement("div");
    messageEl.className = "flex items-start gap-3 animate-fade-in";

    const parsedMarkdown = marked.parse(answerText);
    const uniqueId = "sources_" + Date.now();

    let sourcesHtml = "";
    if (sources && sources.length > 0) {
        const chipsHtml = sources.map((s) => `
            <div class="px-3 py-2 rounded-xl bg-zinc-50 border border-zinc-200 text-xs text-zinc-700 flex items-center justify-between shadow-xs">
                <div class="flex items-center gap-2 overflow-hidden">
                    <span class="font-semibold text-orange-600 px-1.5 py-0.5 rounded bg-orange-500/10 border border-orange-500/20 text-[10px]">Page ${s.page}</span>
                    <span class="truncate text-zinc-800 font-medium">${escapeHtml(s.heading)}</span>
                </div>
                ${s.rerank_score !== undefined ? `<span class="text-[10px] px-2 py-0.5 rounded bg-white text-orange-600 border border-zinc-200 font-mono ml-2">Score: ${s.rerank_score}</span>` : ''}
            </div>
        `).join("");

        sourcesHtml = `
            <div class="mt-3 pt-3 border-t border-zinc-100">
                <button onclick="toggleAccordion('${uniqueId}')" class="text-xs text-zinc-500 hover:text-orange-600 flex items-center gap-1.5 font-medium focus:outline-none transition-colors">
                    <i id="${uniqueId}_icon" class="fa-solid fa-chevron-right text-[10px] transition-transform"></i>
                    <span>Retrieved Context (${sources.length} sources)</span>
                </button>
                <div id="${uniqueId}" class="hidden mt-2 space-y-1.5">
                    ${chipsHtml}
                </div>
            </div>
        `;
    }

    messageEl.innerHTML = `
        <div class="w-8 h-8 rounded-xl bg-white border border-zinc-200 flex items-center justify-center text-sm text-orange-500 flex-shrink-0 mt-1 shadow-xs">
            ${bookIcon || '📖'}
        </div>
        <div class="flex-1 bg-white border border-zinc-200 rounded-2xl rounded-tl-sm p-4 text-zinc-900 shadow-xs">
            <div class="flex items-center justify-between pb-2 mb-2 border-b border-zinc-100 text-xs text-zinc-500">
                <span class="font-bold text-zinc-900">${escapeHtml(bookTitle)}</span>
                <span class="text-[10px] px-2 py-0.5 rounded bg-orange-500/10 text-orange-600 border border-orange-500/20 font-mono">Grounded RAG</span>
            </div>
            <div class="prose max-w-none">
                ${parsedMarkdown}
            </div>
            ${sourcesHtml}
        </div>
    `;

    list.appendChild(messageEl);
    scrollToBottom();
}

function showTypingIndicator(icon) {
    const list = document.getElementById("messagesList");
    const typingId = "typing_" + Date.now();
    const messageEl = document.createElement("div");
    messageEl.id = typingId;
    messageEl.className = "flex items-start gap-3 animate-fade-in";
    messageEl.innerHTML = `
        <div class="w-8 h-8 rounded-xl bg-white border border-zinc-200 flex items-center justify-center text-sm text-orange-500 flex-shrink-0 mt-1 shadow-xs">
            ${icon || '📖'}
        </div>
        <div class="bg-white border border-zinc-200 rounded-2xl rounded-tl-sm px-4 py-3 text-zinc-500 text-sm flex items-center gap-2 shadow-xs">
            <span class="text-xs font-medium text-zinc-600">Searching & Reranking</span>
            <span class="w-1.5 h-1.5 rounded-full bg-orange-500 dot-1"></span>
            <span class="w-1.5 h-1.5 rounded-full bg-orange-500 dot-2"></span>
            <span class="w-1.5 h-1.5 rounded-full bg-orange-500 dot-3"></span>
        </div>
    `;
    list.appendChild(messageEl);
    scrollToBottom();
    return typingId;
}

function removeTypingIndicator(typingId) {
    const el = document.getElementById(typingId);
    if (el) el.remove();
}

function appendErrorMessage(errorMsg) {
    const list = document.getElementById("messagesList");
    const messageEl = document.createElement("div");
    messageEl.className = "flex items-center gap-2 p-3.5 rounded-xl bg-white border border-zinc-200 text-orange-600 text-xs animate-fade-in shadow-xs";
    messageEl.innerHTML = `
        <i class="fa-solid fa-circle-exclamation text-orange-500 text-sm"></i>
        <span>${escapeHtml(errorMsg)}</span>
    `;
    list.appendChild(messageEl);
    scrollToBottom();
}

function toggleAccordion(id) {
    const container = document.getElementById(id);
    const icon = document.getElementById(id + "_icon");
    if (!container) return;

    if (container.classList.contains("hidden")) {
        container.classList.remove("hidden");
        if (icon) icon.style.transform = "rotate(90deg)";
    } else {
        container.classList.add("hidden");
        if (icon) icon.style.transform = "rotate(0deg)";
    }
}

function scrollToBottom() {
    const chatContainer = document.getElementById("chatContainer");
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
