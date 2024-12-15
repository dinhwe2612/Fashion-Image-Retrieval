import streamlit as st
import random

# Mock backend response generator (replace with your actual chatbot logic)
def get_response(user_input):
    responses = [
        "That's interesting! Tell me more.",
        "I'm here to help. What's your next question?",
        "Could you please clarify that?",
        "Let me think... Ah, here it is!",
        "That’s a great question!",
    ]
    return random.choice(responses)  # Replace with your model's response

# Streamlit Chatbox UI
def main():
    st.title("💬 Chatbox with Streamlit")
    st.write("Welcome! Ask me anything, and I'll do my best to respond.")

    # Session State to manage conversation history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous messages
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # User input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Append user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Get and append bot response
        response = get_response(user_input)
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
