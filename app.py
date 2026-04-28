import os
import re
from typing import List, Tuple
import gradio as gr
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from pdf2image import convert_from_path
from PIL import Image

from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# -----------------------------
# Hardware & Quantization Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4-bit Quantization config to prevent GPU Out-of-Memory crashes
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# -----------------------------
# Hugging Face Model Hub IDs
# -----------------------------
VISION_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
EMBED_MODEL_ID = "nlpaueb/legal-bert-base-uncased"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Global Model Cache (Lazy Loading to save memory)
MODELS = {}

def load_models_lazily():
    """Loads models into VRAM only when needed."""
    if "embedder" not in MODELS:
        print("Loading Legal-BERT...")
        MODELS["embed_tok"] = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
        MODELS["embedder"] = AutoModel.from_pretrained(EMBED_MODEL_ID).to(DEVICE).eval()
        
    if "vision" not in MODELS:
        print("Loading Qwen2-VL (4-bit)...")
        MODELS["vision_proc"] = AutoProcessor.from_pretrained(VISION_MODEL_ID)
        MODELS["vision"] = Qwen2VLForConditionalGeneration.from_pretrained(
            VISION_MODEL_ID, 
            quantization_config=quantization_config,
            device_map="auto"
        ).eval()

    if "llm" not in MODELS:
        print("Loading Qwen2.5 LLM (4-bit)...")
        MODELS["llm_tok"] = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        MODELS["llm"] = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto"
        ).eval()

# -----------------------------
# Step 1: Vision Ingestion Layer
# -----------------------------
import os
import math

# -----------------------------
# Step 1: PARALLEL Vision Ingestion Layer
# -----------------------------
def process_pdf_with_hf_vision(file_path: str) -> str:
    if not file_path: return ""
    load_models_lazily()
    
    try:
        # PARALLEL UPGRADE 1: CPU Multi-threading for Rasterization
        # This utilizes all your CPU cores to rip the PDF into images instantly.
        cpu_threads = os.cpu_count() or 4
        images = convert_from_path(file_path, dpi=150, thread_count=cpu_threads) 
        # Note: Dropped DPI to 150. It's plenty for large 12pt font and saves massive VRAM during batching.
        
        extracted_texts = []
        proc = MODELS["vision_proc"]
        vlm = MODELS["vision"]

        # PARALLEL UPGRADE 2: GPU Mini-Batching
        # Process 3 to 4 pages simultaneously. If you get an OOM error, lower this to 2.
        # If you have a 24GB GPU (like an RTX 3090/4090), you can push this to 6 or 8.
        BATCH_SIZE = 3 
        
        for i in range(0, len(images), BATCH_SIZE):
            batch_imgs = images[i:i+BATCH_SIZE]
            print(f"Processing pages {i+1} to {i+len(batch_imgs)} in parallel...")
            
            # Prepare the batched prompts
            batch_texts = []
            for img in batch_imgs:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Extract all text, tables, and logical structure from this document page exactly as it appears. Output plain text."},
                        ],
                    }
                ]
                text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_texts.append(text)
            
            # Process the whole batch into tensors at once
            inputs = proc(
                text=batch_texts, 
                images=batch_imgs, 
                padding=True, 
                return_tensors="pt"
            ).to(DEVICE)

            # Run parallel inference
            with torch.inference_mode():
                generated_ids = vlm.generate(**inputs, max_new_tokens=1024)
                
                # Trim the prompt tokens off the output for the whole batch
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                # Decode all pages in the batch simultaneously
                batch_pages_text = proc.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
                extracted_texts.extend(batch_pages_text)
            
            # CRITICAL: Manually clear the GPU cache after each batch to prevent memory leaks
            torch.cuda.empty_cache()
            
        return "\n".join(extracted_texts)
    except Exception as e:
        return f"Vision Processing Error: {str(e)}"
# -----------------------------
# Step 2: Extractive Logic (MMR Bottleneck)
# -----------------------------
def simple_sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    candidates = re.split(r"(?<=[.!?])\s+", text)
    merged, buf = [], ""
    for c in candidates:
        frag = c.strip()
        if not frag: continue
        if not buf: buf = frag
        else:
            if len(frag) <= 3 and re.match(r"^[\(\)\[\]\dA-Za-z\-:;+.,]+$", frag):
                buf += " " + frag
            else:
                merged.append(buf)
                buf = frag
    if buf: merged.append(buf)
    return [s.strip() for s in merged if s.strip()]

@torch.inference_mode()
def embed_sentences(sentences: List[str]) -> np.ndarray:
    tok = MODELS["embed_tok"]
    mdl = MODELS["embedder"]
    vecs = []
    for i in range(0, len(sentences), 32):
        batch = sentences[i:i+32]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        out = mdl(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = torch.sum(out.last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        sent_emb = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
        vecs.append(sent_emb.cpu().numpy())
    return np.vstack(vecs)

def mmr_select(sentence_embeddings: np.ndarray, top_n: int, lambda_param: float = 0.5) -> List[int]:
    if len(sentence_embeddings) <= top_n: return list(range(len(sentence_embeddings)))
    
    doc_embedding = np.mean(sentence_embeddings, axis=0).reshape(1, -1)
    unselected = list(range(len(sentence_embeddings)))
    selected = []

    while len(selected) < top_n and unselected:
        if not selected:
            sims = cosine_similarity(sentence_embeddings[unselected], doc_embedding)
            best_idx = unselected[np.argmax(sims)]
        else:
            sim_to_doc = cosine_similarity(sentence_embeddings[unselected], doc_embedding)
            sim_to_selected = cosine_similarity(sentence_embeddings[unselected], sentence_embeddings[selected])
            max_sim_to_selected = np.max(sim_to_selected, axis=1).reshape(-1, 1)
            mmr_scores = lambda_param * sim_to_doc - (1 - lambda_param) * max_sim_to_selected
            best_idx = unselected[np.argmax(mmr_scores)]
            
        selected.append(best_idx)
        unselected.remove(best_idx)

    return sorted(selected)

# -----------------------------
# Step 3: Generative LLM Layer
# -----------------------------
@torch.inference_mode()
def run_hf_llm(text: str) -> str:
    tok = MODELS["llm_tok"]
    llm = MODELS["llm"]
    
    messages = [
        {"role": "system", "content": "You are an expert quantitative legal clerk. Synthesize the following document core into a highly professional summary strictly between 250 and 500 words. Do not output anything other than the summary."},
        {"role": "user", "content": f"DOCUMENT CORE:\n{text}"}
    ]
    
    # Use chat template for modern instruct models
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
    
    # Parameters tuned for cohesive 250-500 word output
    gen = llm.generate(
        **inputs,
        min_new_tokens=350,
        max_new_tokens=700,
        repetition_penalty=1.1,
        temperature=0.3, # Low temp for factual legal consistency
        do_sample=True
    )
    
    # Strip prompt from output
    output_ids = gen[0][inputs.input_ids.shape[1]:]
    return tok.decode(output_ids, skip_special_tokens=True)

# -----------------------------
# Master Pipeline
# -----------------------------
def pipeline(pdf_file, pasted_text: str, num_sentences: float) -> Tuple[str, str, str]:
    num_sentences = int(num_sentences)
    
    if pdf_file is not None:
        target_text = process_pdf_with_hf_vision(pdf_file)
        if "Error" in target_text: return "", "", target_text
    else:
        target_text = pasted_text

    if not target_text.strip(): return "", "", "No input provided."
    load_models_lazily()

    sentences = simple_sentence_split(target_text)
    if not sentences: return "", "", "Failed to split text."
         
    embs = embed_sentences(sentences)
    chosen_idx = mmr_select(embs, top_n=num_sentences, lambda_param=0.6)
    extractive_core = " ".join([sentences[i] for i in chosen_idx])

    try:
        summary = run_hf_llm(extractive_core)
    except Exception as e:
        return "", f"Generation Error: {str(e)}", ""

    word_count = len(summary.split())
    raw_word_count = len(target_text.split())
    
    dbg = (
        f"Pipeline: Pure Hugging Face (4-bit Quantized)\n"
        f"Raw Words Parsed: ~{raw_word_count}\n"
        f"Extracted Core: {len(chosen_idx)} golden sentences.\n"
        f"Final Word Count: ~{word_count} words."
    )
    return extractive_core, summary, dbg

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Hybrid Vision-LED Summarizer") as demo:
    gr.Markdown("# 👁️‍🗨️ 100% Hugging Face Multimodal Summarizer")
    gr.Markdown("Powered by Qwen-VL (Vision), Legal-BERT (Vector Bottleneck), and Qwen2.5 (LLM Generation). All running locally via `transformers`.")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_upload = gr.File(label="1. Upload Scanned PDF (Vision Inference)", file_types=[".pdf"], type="filepath")
            text_in = gr.Textbox(label="1b. Or Paste Text", lines=4)
        with gr.Column(scale=1):
            num_sentences = gr.Slider(label="2. MMR Bottleneck Size (Sentences)", minimum=20, maximum=120, value=50, step=5)
            run_btn = gr.Button("Execute Full Stack 🚀", variant="primary")
    
    with gr.Row():
        abstractive_out = gr.Textbox(label="Final Summary (Strictly 250-500 words)", lines=12)
    with gr.Accordion("Pipeline Engine Diagnostics", open=False):
        debug_out = gr.Textbox(label="Diagnostics", lines=5)
        extractive_out = gr.Textbox(label="Golden Context (Passed from BERT to LLM)", lines=8)

    run_btn.click(
        pipeline, 
        inputs=[pdf_upload, text_in, num_sentences], 
        outputs=[extractive_out, abstractive_out, debug_out]
    )

if __name__ == "__main__":
    # server_name="0.0.0.0" is required for Docker port mapping to work
    demo.launch(server_name="0.0.0.0", server_port=7860)
