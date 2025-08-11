from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

# ====== CONFIG ======
os.environ["GROQ_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Store conversations in memory (per client_id)
chat_sessions = {}

# ====== PROMPT TEMPLATE ======
template = """
You are a friendly and professional hotel booking assistant at Prismatics Hotels.
Speak naturally like a mature human receptionist and avoid repeating the same answers.
Keep replies short (1–2 sentences), polite, and relevant only to hotel bookings.
Ask one question at a time until you have gathered:
1. How you can assist with hotel booking
2. Number of days of stay
3. Number of people
4. Room type (single, double, suite)
5. Budget per night
6. check in date 

Room rates:
- Single: 2000Rs/night
- Double: 3000Rs/night
- Suite: 4000Rs/night

If the budget is slightly lower than the price, offer a discount between 3% and 5% only if the client asks for it.
Decide the discount based on the situation and never offer it upfront.
Politely encourage booking after discussing prices.

When the booking is finalized, give a short confirmation and include:
Location: 71-C Gulberg 3, PC Hotels
Tel: 0333-3333333

Keep the tone friendly, concise, and helpful.

Conversation so far:
{history}
User: {input}
Assistant:
"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# ====== FASTAPI APP ======
app = FastAPI(title="Hotel Booking Chatbot API")

class ChatRequest(BaseModel):
    client_id: str
    message: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    client_id = request.client_id
    user_message = request.message.strip()

    if client_id not in chat_sessions:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
        memory = ConversationBufferMemory(memory_key="history", return_messages=False)
        chat_sessions[client_id] = ConversationChain(prompt=prompt, llm=llm, memory=memory)

    chain = chat_sessions[client_id]
    bot_reply = chain.predict(input=user_message)

    return {
        "client_id": client_id,
        "bot_reply": bot_reply
    }

@app.get("/sessions")
def list_sessions():
    return {"active_sessions": list(chat_sessions.keys())}

