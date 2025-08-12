import os
from datetime import datetime

import streamlit as st
from huggingface_hub import InferenceClient


st.set_page_config(page_title="LLM Playground", page_icon="🧠")
st.title("🧠 LLM Playground")


if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role: "user"|"assistant", content: str, time: str}


with st.sidebar:
    st.subheader("Settings")
    token = st.text_input(
        "Hugging Face API Token",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        help="Paste your Hugging Face access token. You can also set HF_TOKEN env var.",
    )

    model = st.selectbox(
        "Model",
        options=[
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mistral-7B-Instruct-v0.1",
        ],
        index=0,
    )

    max_tokens = st.slider("Max tokens", min_value=128, max_value=1024, value=512, step=64)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Prepare exportable text
    export_lines = ["🧠 LLM Playground conversation export\n"]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "Bot"
        ts = f" [{m.get('time','')}]" if m.get("time") else ""
        export_lines.append(f"{who}{ts}: {m['content']}")
    export_text = "\n\n".join(export_lines)

    st.download_button(
        "⬇️ Export conversation (.txt)",
        data=export_text,
        file_name="conversation.txt",
        mime="text/plain",
        disabled=len(st.session_state.messages) == 0,
    )


# Display chat history
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        # Show timestamp subtly above the message if available
        if message.get("time"):
            st.caption(message["time"]) 
        st.markdown(message["content"])  # supports markdown formatting


prompt = st.chat_input("Type your message and press Enter…")
if prompt:
    # Add user message to history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt.strip(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    # Echo it in the UI
    with st.chat_message("user", avatar="🧑"):
        st.caption(st.session_state.messages[-1]["time"]) 
        st.markdown(prompt)

    # Use token from sidebar or environment variable
    cleaned_token = (token or os.getenv("HF_TOKEN", "")).strip()

    if not cleaned_token:
        with st.chat_message("assistant", avatar="🤖"):
            st.warning("Please provide your Hugging Face token in the sidebar to continue.")
    else:
        # Construct messages for the HF InferenceClient, including a system prompt
        messages_for_model = [
            {"role": "system", "content": "You are a helpful and knowledgeable assistant."}
        ]
        for m in st.session_state.messages:
            messages_for_model.append({"role": m["role"], "content": m["content"]})

        # Try different client configurations
        try:
            # First try with token
            client = InferenceClient(model=model, token=cleaned_token)
        except Exception:
            # Fallback: try without token (for public models)
            client = InferenceClient(model=model)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                try:
                    # Try chat completion first
                    response = client.chat_completion(
                        messages=messages_for_model,
                        max_tokens=max_tokens,
                    )
                    answer = response.choices[0].message.content.strip()
                except Exception as e:
                    # Fallback to text generation with simple prompt
                    try:
                        # Build a simple prompt for text generation
                        user_messages = [m["content"] for m in messages_for_model if m["role"] == "user"]
                        if user_messages:
                            latest_user_message = user_messages[-1]
                            prompt_text = f"<s>[INST] {latest_user_message} [/INST]"
                            
                            tg = client.text_generation(
                                prompt_text,
                                max_new_tokens=max_tokens,
                                do_sample=True,
                                temperature=0.7,
                            )
                            answer = (tg or "").strip()
                            if not answer:
                                raise RuntimeError("Empty response from text_generation")
                        else:
                            raise RuntimeError("No user message found")
                    except Exception as fallback_error:
                        answer = f"Error: {e}\nFallback also failed: {fallback_error}"

            st.markdown(answer)

        # Save assistant reply
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )


