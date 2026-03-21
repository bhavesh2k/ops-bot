const input = document.getElementById("question");
const chat = document.getElementById("chat");
const sourceList = document.getElementById("sourceList");
const sendBtn = document.querySelector(".send-btn");


function addCopyButtons(container) {
  const blocks = container.querySelectorAll("pre");

  blocks.forEach(block => {
    // جلوگیری duplicate button
    if (block.parentElement.classList.contains("code-wrapper")) return;

    const wrapper = document.createElement("div");
    wrapper.className = "code-wrapper";

    const button = document.createElement("button");
    button.className = "copy-btn";
    button.innerText = "Copy";

    button.onclick = () => {
      const code = block.innerText;
      navigator.clipboard.writeText(code);

      button.innerText = "Copied!";
      setTimeout(() => {
        button.innerText = "Copy";
      }, 1500);
    };

    block.parentNode.insertBefore(wrapper, block);
    wrapper.appendChild(button);
    wrapper.appendChild(block);
  });
}


/* ENTER KEY */
input.addEventListener("keypress", function (event) {
  if (event.key === "Enter") {
    sendQuestion();
  }
});

/* SEND BUTTON */
sendBtn.addEventListener("click", sendQuestion);

async function sendQuestion() {

  let fullResponse = "";

  const question = input.value.trim();
  if (!question) return;

  input.value = "";

  /* CLEAR SOURCES */
  sourceList.innerHTML = "";

  /* USER MESSAGE */

  const userMessage = document.createElement("div");
  userMessage.className = "message user";

  userMessage.innerHTML = `
      <div class="user-icon">🧑‍💻</div>
      <div class="bubble">${question}</div>
  `;

  chat.appendChild(userMessage);

  /* BOT MESSAGE */

  const botMessage = document.createElement("div");
  botMessage.className = "message";

  botMessage.innerHTML = `
      <div class="bot-icon">🤖</div>
      <div class="bubble"><span class="thinking">Thinking...</span></div>
  `;

  chat.appendChild(botMessage);

  const botBubble = botMessage.querySelector(".bubble");
  let hasReceivedFirstToken = false;

  let dots = 0;
  const thinkingInterval = setInterval(() => {
    if (hasReceivedFirstToken) {
      clearInterval(thinkingInterval);
      return;
    }
    dots = (dots + 1) % 4;
    botBubble.innerHTML = `<span class="thinking">Thinking${".".repeat(dots)}</span>`;
  }, 400);

  chat.scrollTop = chat.scrollHeight;

  const response = await fetch("/ask/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ question: question })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value);

    let lines = buffer.split("\n");

    // keep last incomplete chunk
    buffer = lines.pop();

    for (let line of lines) {
      if (!line.trim()) continue;

      try {
        const event = JSON.parse(line);

        if (event.type === "sources") {
          sourceList.innerHTML = ""; // clear default

          event.data.forEach(src => {
            const li = document.createElement("li");
            li.innerText = src;
            sourceList.appendChild(li);
          });
        }

        if (event.type === "start_answer") {
          // DO NOTHING yet — keep "Thinking..." visible
        }

        if (event.type === "token") {

          if (!hasReceivedFirstToken) {
            botBubble.innerHTML = "";
            hasReceivedFirstToken = true;
          }

          fullResponse += event.data;

          // Render markdown live
          botBubble.innerHTML = marked.parse(fullResponse);
          
          // Add copy buttons after rendering markdown
          addCopyButtons(botBubble);
        }

      } catch (e) {
        console.error("Parsing error:", e, line);
      }
    }

    chat.scrollTop = chat.scrollHeight;
  }
}