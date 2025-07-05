const elLog  = document.getElementById("chat-container");
const elForm = document.getElementById("chat-form");
const elBox  = document.getElementById("chat-input");

function addBubble(text, cls){
  const div=document.createElement("div");
  div.className=`msg ${cls}`;
  div.textContent=text;
  elLog.appendChild(div);
  elLog.scrollTop=elLog.scrollHeight;
  return div;
}

elForm.addEventListener("submit", e=>e.preventDefault());
elBox.addEventListener("keydown", e=>{
  if((e.key==="Enter" && !e.shiftKey) || (e.key==="Enter" && e.ctrlKey)){
    e.preventDefault();
    send();
  }
});

async function send(){
  const q=elBox.value.trim(); if(!q) return;
  elBox.value=""; addBubble(q,"user");
  const aiDiv=addBubble("…","ai");

  const res=await fetch("/chat_stream",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({prompt:q,max_tokens:256})});
  const reader=res.body.getReader(); const decoder=new TextDecoder();
  let buf="";
  while(true){
    const {value,done}=await reader.read(); if(done) break;
    buf+=decoder.decode(value,{stream:true});
    for(const chunk of buf.split("\n\n")){
      if(chunk.startsWith("data: ")){const tok=chunk.slice(6).trim();
        if(tok==="[DONE]") return;
        aiDiv.textContent += (aiDiv.textContent==="…"?"": " ")+tok;
        elLog.scrollTop=elLog.scrollHeight;
      }
    }
    buf="";
  }
}