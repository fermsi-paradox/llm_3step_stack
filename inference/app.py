import gradio as gr
import torch
import os
import threading
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
try:
    # Use Rust-powered downloader for more reliable shard downloads in Spaces
    from huggingface_hub import hf_hub_enable_hf_transfer
    hf_hub_enable_hf_transfer()
except Exception:
    pass
import traceback

# Use your fine-tuned model (replace with your model name)
MODEL_NAME = os.environ.get("MODEL_NAME", "your-username/your-model-name")

model = None
tokenizer = None
loading_status = {"is_loading": False}

def load_model():
    global model, tokenizer, loading_status
    
    if loading_status["is_loading"]:
        return False
        
    loading_status["is_loading"] = True
    print("🤖 Loading fine-tuned model with 4-bit quantization...")
    
    try:
        # Configure quantization properly for Mixtral
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # Better for Mixtral
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Optional auth token for private models
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        # Improve reliability of hub downloads in Spaces
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("HF_HUB_ENABLE_TELEMETRY", "0")
        model_load_kwargs = dict(
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            resume_download=True,
        )
        if hf_token:
            model_load_kwargs["token"] = hf_token

        # Load with retries to handle transient S3/CDN hiccups
        last_err = None
        for attempt in range(1, 4):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    **model_load_kwargs,
                )
                tokenizer_load_kwargs = dict(trust_remote_code=True, resume_download=True)
                if hf_token:
                    tokenizer_load_kwargs["token"] = hf_token
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tokenizer_load_kwargs)
                last_err = None
                break
            except Exception as e:
                last_err = e
                wait_s = 2 ** attempt
                print(f"⚠️ Download attempt {attempt} failed: {e}. Retrying in {wait_s}s...")
                time.sleep(wait_s)
        if last_err is not None:
            raise last_err
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded successfully with 4-bit quantization!")
        loading_status["is_loading"] = False
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        loading_status["is_loading"] = False
        return False

def generate_response(message, history):
    global model, tokenizer, loading_status
    
    # Check if model is loaded
    if model is None:
        if loading_status["is_loading"]:
            return "🤖 Model is currently loading... Please wait a moment and try again."
        
        # Show immediate loading feedback
        loading_message = "🤖 Loading fine-tuned model for the first time... This may take 3-5 minutes due to the large model size. Please be patient!"
        
        load_success = load_model()
        if not load_success:
            return "❌ Model failed to load. Please try again in a few minutes or contact support. This might be a temporary HuggingFace issue or the model upload may not have completed successfully."
        
        # Model loaded successfully, now generate response
        return f"✅ Model loaded successfully! Now processing your question: '{message}'\n\n" + _generate_actual_response(message, history)
    
    return _generate_actual_response(message, history)

def _generate_actual_response(message, history):
    global model, tokenizer
    
    try:
        def needs_continuation(text: str) -> bool:
            if not text:
                return False
            tail = text.strip()[-1:]
            ended = tail in ".!?"
            return not ended

        # Simple anti-loop guard: if the last assistant message repeats the same first sentence
        # as the current draft, force a concise re-generation with different seed/temperature
        def looks_repetitive(draft: str, history_pairs) -> bool:
            if not draft or not history_pairs:
                return False
            last_pair = history_pairs[-1]
            if len(last_pair) < 2 or not last_pair[1]:
                return False
            prev = last_pair[1].split(".\n")[0][:120]
            cur = draft.split(".\n")[0][:120]
            return prev.strip().lower() == cur.strip().lower()

        # Create a proper prompt format for Mixtral Instruct
        expert_prompt = """
# SYSTEM PROMPT PLACEHOLDER

Replace this placeholder with your custom system prompt that defines:
- The assistant's role and expertise area
- Key behavioral guidelines and constraints  
- Response formatting preferences
- Domain-specific knowledge and facts
- Default settings and configurations
- Safety considerations and requirements
- Contact information and support details

This system prompt should be comprehensive and tailored to your specific use case.
"""

        # System message MUST be first, then alternate user/assistant
        messages = [{"role": "system", "content": expert_prompt}]

        # Add conversation history pairs (ChatInterface provides (user, assistant))
        # Cap to the last N exchanges to avoid conversation drift/echo loops
        max_history_pairs = 6
        capped_history = history[-max_history_pairs:] if history else []
        for user_msg, assistant_msg in capped_history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current user message as the last message before generation
        messages.append({"role": "user", "content": message})
        
        # Apply chat template
        conversation = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Let the model handle all responses based on the system prompt
        # No hardcoded conditional logic - the fine-tuned model should handle domain-specific responses
        
        # Tokenize input
        inputs = tokenizer(
            conversation, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096  # Mixtral can handle longer contexts
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # Generate response with Mixtral-optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=420,
                temperature=0.5,
                do_sample=True,
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "[/INST]" in full_response:
            # Mixtral format
            response = full_response.split("[/INST]")[-1].strip()
        elif "<|im_start|>assistant" in full_response:
            # Alternative format
            response = full_response.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
        else:
            # Fallback: remove the input conversation
            response = full_response[len(conversation):].strip()
        
        # Clean up any artifacts
        response = response.replace("<|endoftext|>", "").strip()
        response = response.replace("</s>", "").strip()

        # If the response appears identical to the prior assistant message, make a brief re-try
        if looks_repetitive(response, capped_history):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=320,
                    temperature=0.6,
                    do_sample=True,
                    top_p=0.9,
                    top_k=60,
                    repetition_penalty=1.25,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()
            elif "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
            else:
                response = response[len(conversation):].strip()
            response = response.replace("<|endoftext|>", "").replace("</s>", "").strip()
        
        # If response likely cut off, request a short continuation to complete steps
        if needs_continuation(response):
            continuation_messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content": "Continue the previous answer concisely. Finish any remaining numbered steps."},
            ]
            continuation_conv = tokenizer.apply_chat_template(
                continuation_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            cont_inputs = tokenizer(
                continuation_conv,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            if torch.cuda.is_available():
                cont_inputs = cont_inputs.to("cuda")
            with torch.no_grad():
                cont_outputs = model.generate(
                    **cont_inputs,
                    max_new_tokens=160,
                    temperature=0.45,
                    do_sample=True,
                    top_p=0.85,
                    top_k=30,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            cont_text = tokenizer.decode(cont_outputs[0], skip_special_tokens=True)
            if "[/INST]" in cont_text:
                cont_text = cont_text.split("[/INST]")[-1].strip()
            elif "<|im_start|>assistant" in cont_text:
                cont_text = cont_text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
            else:
                cont_text = cont_text[len(continuation_conv):].strip()
            cont_text = cont_text.replace("<|endoftext|>", "").replace("</s>", "").strip()
            response = (response + "\n" + cont_text).strip()
        
        print(f"Generated response: {response[:100]}...")  # Debug log
        
        return response
        
    except Exception as e:
        error_msg = f"Generation error: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"Full traceback: {traceback.format_exc()}")
        return f"Sorry, I encountered an error generating a response: {error_msg}"

# Get app configuration from environment variables
APP_TITLE = os.environ.get("APP_TITLE", "Expert AI Assistant")
APP_DESCRIPTION = os.environ.get("APP_DESCRIPTION", "AI assistant powered by fine-tuned language model")
APP_THEME_COLOR = os.environ.get("APP_THEME_COLOR", "#3b82f6")  # Default blue

with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
    # Customizable header
    gr.HTML(
        f"""
        <style>
          .header-wrap {{ display:flex; align-items:center; justify-content:center; padding: 16px 0 6px; }}
          .header-wrap h1 {{
            margin:0; font-size: 28px; font-weight: 800; letter-spacing: 0.4px;
            color: {APP_THEME_COLOR};
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
            text-align:center;
          }}
          .header-wrap p {{
            margin: 6px 0 0; color: #9ca3af; font-size: 13px; text-align: center;
          }}
        </style>
        <div class="header-wrap">
          <div>
            <h1>{APP_TITLE}</h1>
            <p>{APP_DESCRIPTION}</p>
          </div>
        </div>
        """
    )
    
    gr.ChatInterface(
        generate_response
    )

    # Warm start: pre-load model in background when the Space boots to avoid first-request delay
    try:
        threading.Thread(target=load_model, daemon=True).start()
    except Exception:
        # If background preload fails, the model will still load on first request
        pass

if __name__ == "__main__":
    demo.launch() 