const input = document.getElementById("question");

input.addEventListener("keypress", function(event){
    if(event.key === "Enter"){
        sendQuestion();
    }
});

async function sendQuestion(){

    const chat = document.getElementById("chat");
    const sourceList = document.getElementById("sourceList");

    const question = input.value.trim();

    if(!question) return;

    input.value = "";

    sourceList.innerHTML = "";

    /* USER MESSAGE */

    const userMessage = document.createElement("div");
    userMessage.className = "message user-message";

    userMessage.innerHTML = `
        <div class="bubble user-bubble">${question}</div>
    `;

    chat.appendChild(userMessage);

    /* ASSISTANT MESSAGE */

    const botMessage = document.createElement("div");
    botMessage.className = "message bot-message";

    botMessage.innerHTML = `
        <div class="bubble bot-bubble thinking">Thinking...</div>
    `;

    chat.appendChild(botMessage);

    const botBubble = botMessage.querySelector(".bubble");

    chat.scrollTop = chat.scrollHeight;

    const response = await fetch("/ask/stream", {
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({question:question})
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";
    let inSources = false;
    let inAnswer = false;
    let firstAnswerToken = true;

    while(true){

        const {value, done} = await reader.read();

        if(done) break;

        const chunk = decoder.decode(value);

        buffer += chunk;

        if(buffer.includes("Sources:")){
            inSources = true;
            buffer = buffer.replace("Sources:", "");
        }

        if(buffer.includes("Answer:")){
            inSources = false;
            inAnswer = true;
            buffer = buffer.replace("Answer:", "");            
        }

        if(inSources){
            const lines = buffer.split("\n");
            lines.forEach(line=>{
                if(line.trim().startsWith("-")){
                    const li = document.createElement("li");
                    li.innerText = line.replace("-","").trim();
                    sourceList.appendChild(li);
                }
            });

            buffer = "";
        }

        if (inAnswer) {
            if (buffer.trim().length > 0) {  // only act when buffer has real content
                if (firstAnswerToken) {
                    botBubble.classList.remove("thinking"); // remove "Thinking..." now
                    botBubble.innerText = "";             // clear bubble
                    firstAnswerToken = false;
                }

                botBubble.innerText += buffer;
                buffer = "";
            }
        }

        chat.scrollTop = chat.scrollHeight;
    }
}