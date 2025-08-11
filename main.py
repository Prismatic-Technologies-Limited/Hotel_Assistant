import json
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatGroq

# Initialize Groq LLM
llm = ChatGroq(model="mixtral-8x7b-32768", api_key="YOUR_GROQ_API_KEY")

# Tools
@tool
def get_hotel_info_tool(query: str) -> str:
    """Returns hotel information."""
    hotel_info = """
    Welcome to AI Grand Hotel!
    - Location: Beach Road, Goa
    - Amenities: Pool, Spa, Gym, Free WiFi
    - Check-in: 2 PM, Check-out: 11 AM
    """
    return hotel_info

@tool
def check_room_availability_tool(room_type: str) -> str:
    """Checks if a room is available."""
    with open("rooms.json") as f:
        rooms = json.load(f)["rooms"]
    available_rooms = [r for r in rooms if r["type"].lower() == room_type.lower() and r["available"]]
    if available_rooms:
        return f"Room available: {available_rooms[0]}"
    return "No rooms of this type are available."

@tool
def reserve_room_tool(room_type: str, name: str) -> str:
    """Reserves a room if available."""
    with open("rooms.json") as f:
        rooms = json.load(f)
    for room in rooms["rooms"]:
        if room["type"].lower() == room_type.lower() and room["available"]:
            room["available"] = False
            with open("rooms.json", "w") as fw:
                json.dump(rooms, fw)
            return f"Room {room['room_number']} reserved for {name} at ${room['price']}."
    return "Reservation failed: No available room."

@tool
def negotiate_price_tool(room_type: str, requested_discount: float) -> str:
    """Negotiates price up to 7% discount."""
    with open("rooms.json") as f:
        rooms = json.load(f)["rooms"]
    for room in rooms:
        if room["type"].lower() == room_type.lower():
            max_discount = 7
            discount = min(requested_discount, max_discount)
            new_price = room["price"] * (1 - discount / 100)
            return f"Best offer: ${new_price:.2f} (Discount applied: {discount}%)"
    return "Room type not found."

# Prompt
template = """
You are an AI hotel receptionist.
Answer politely, provide hotel info, check availability, reserve rooms, and negotiate price (max 7% discount).
Conversation:
{history}
User: {input}
AI:
"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
memory = ConversationBufferMemory(memory_key="history")
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Example conversation
print(chain.run(input="Tell me about your hotel"))
print(chain.run(input="I want to book a Deluxe room"))
print(chain.run(input="Can you give me 10% off?"))
