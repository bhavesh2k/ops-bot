const input = document.getElementById("question");
const chat = document.getElementById("chat");
const sourceList = document.getElementById("sourceList");
const sendBtn = document.querySelector(".send-btn");

/* ENTER KEY */
input.addEventListener("keypress", function (event) {
  if (event.key === "Enter") {
    sendQuestion();
  }
});

/* SEND BUTTON */
sendBtn.addEventListener("click", sendQuestion);

async function sendQuestion() {

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
      <div class="bubble">Thinking...</div>
  `;

  chat.appendChild(botMessage);

  const botBubble = botMessage.querySelector(".bubble");

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
  let inSources = false;
  let inAnswer = false;

  botBubble.innerText = "";

  while (true) {

    const { value, done } = await reader.read();

    if (done) break;

    const chunk = decoder.decode(value);
    buffer += chunk;

    /* SOURCES SECTION */

    if (buffer.includes("Sources:")) {
      inSources = true;
      buffer = buffer.replace("Sources:", "");
    }

    if (buffer.includes("Answer:")) {
      inSources = false;
      inAnswer = true;
      buffer = buffer.replace("Answer:", "");
    }

    if (inSources) {

      const lines = buffer.split("\n");

      lines.forEach(line => {

        if (line.trim().startsWith("-")) {

          const li = document.createElement("li");
          li.innerText = line.replace("-", "").trim();

          sourceList.appendChild(li);
        }

      });

      buffer = "";
    }

    /* ANSWER STREAM */

    if (inAnswer) {
      botBubble.innerText += buffer;
      buffer = "";
    }

    chat.scrollTop = chat.scrollHeight;
  }
}