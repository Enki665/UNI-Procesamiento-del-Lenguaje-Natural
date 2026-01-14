import torch
import gc
import time
import csv
import os
import numpy as np
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer


archivo_dataset = "dataset_entrenamiento.jsonl"
archivo_reporte = "REPORTE_METRICAS_3_EPOCAS.csv"
max_seq_length = 2048
load_in_4bit = True


lista_modelos = [
    {"nombre": "unsloth/llama-3-8b-Instruct-bnb-4bit", "carpeta": "modelo_final_llama3", "bs": 2, "ga": 4},
    {"nombre": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", "carpeta": "modelo_final_mistral", "bs": 2, "ga": 4},
    {"nombre": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "carpeta": "modelo_final_qwen", "bs": 2, "ga": 4},
    {"nombre": "unsloth/gemma-2-9b-it-bnb-4bit", "carpeta": "modelo_final_gemma", "bs": 1, "ga": 8}
]

test_prompt_text = "El titular del proyecto deberá asegurar los reportes de incidentes ambientales de manera mensual."
respuesta_ideal = "Tipo: Obligación Fiscalizable\nCategoría: Residuos Peligrosos\nTema: Transporte autorizado"
system_prompt_text = "Eres un experto en fiscalización ambiental y cumplimiento legal."



print("Cargando jueces de calidad (MiniLM & ROUGE)...")
evaluator_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') # CPU para no gastar VRAM
rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def limpiar_memoria():
    print("Limpiando memoria GPU...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

encabezados = [
    "Modelo", "Tiempo Train (s)", "Loss Final", "VRAM (GB)", 
    "Velocidad (tok/s)", "Similitud Semántica (0-1)", "ROUGE-L Score"
]
if not os.path.isfile(archivo_reporte):
    with open(archivo_reporte, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(encabezados)

# entrenamiento
for config in lista_modelos:
    model_name = config["nombre"]
    output_dir = config["carpeta"]
    bs = config["bs"]
    ga = config["ga"]
    
    print(f"\n{'='*60}")
    print(f"PROCESANDO: {model_name}")
    print(f"Objetivo: 3 Épocas | Batch Size: {bs}")
    print(f"{'='*60}\n")
    
    limpiar_memoria()
    
    # inicializar variables
    model = None
    tokenizer = None
    trainer = None
    
    try:
        # cargar Modelo Base
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = load_in_4bit,
        )

        # configurar Adaptadores (LoRA)
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )

        # preparación Inteligente de Datos
        dataset = load_dataset("json", data_files=archivo_dataset, split="train")
        
        # detectar si es Gemma (La problemática)
        is_gemma = "gemma" in model_name.lower()
        
        def formatting_prompts_func(examples):
            texts = []
            for convo in examples["messages"]:
                if is_gemma:

                    sys = next((m['content'] for m in convo if m['role'] == 'system'), "")
                    usr = next((m['content'] for m in convo if m['role'] == 'user'), "")
                    ast = next((m['content'] for m in convo if m['role'] == 'assistant'), "")

                    new_convo = [
                        {"role": "user", "content": f"{sys}\n\nInstrucción: {usr}"},
                        {"role": "assistant", "content": ast}
                    ]
                    texts.append(tokenizer.apply_chat_template(new_convo, tokenize=False, add_generation_prompt=False))
                else:

                    texts.append(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))
            return { "text" : texts }

        dataset = dataset.map(formatting_prompts_func, batched = True)

        # entrenamiento
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False,
            args = TrainingArguments(
                per_device_train_batch_size = bs,
                gradient_accumulation_steps = ga,
                warmup_steps = 10,
                num_train_epochs = 3, 
                learning_rate = 2e-4,
                fp16 = False,
                bf16 = True, 
                logging_steps = 10,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = output_dir,
            ),
        )
        
        print("Iniciando entrenamiento intenso...")
        start_time = time.time()
        trainer_stats = trainer.train()
        training_time = round(time.time() - start_time, 2)
        final_loss = round(trainer_stats.metrics['train_loss'], 4)
        
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)

        print(f"Guardando modelo en {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # evaluación de Calidad (Inferencia)
        print("Evaluando calidad semántica...")
        FastLanguageModel.for_inference(model)
        
        if is_gemma:
            msg = [{"role": "user", "content": f"{system_prompt_text}\n\nAnaliza el siguiente texto:\n{test_prompt_text}"}]
        else:
            msg = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": f"Analiza el siguiente texto:\n{test_prompt_text}"}
            ]
            
        inputs = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        
        t_start = time.time()
        outputs = model.generate(inputs, max_new_tokens=128, use_cache=True, temperature=0.1)
        tokens_gen = len(outputs[0]) - len(inputs[0])
        inf_speed = round(tokens_gen / (time.time() - t_start), 2)
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in full_response:
             respuesta_generada = full_response.split("assistant")[-1].strip()
        elif "model" in full_response:
             respuesta_generada = full_response.split("model")[-1].strip()
        else:
             respuesta_generada = full_response.split("\n")[-1].strip()

        # métricas
        emb1 = evaluator_model.encode(respuesta_generada, convert_to_tensor=True)
        emb2 = evaluator_model.encode(respuesta_ideal, convert_to_tensor=True)
        semantic_score = round(util.pytorch_cos_sim(emb1, emb2).item(), 4)
        
        rouge_scores = rouge_evaluator.score(respuesta_ideal, respuesta_generada)
        rouge_l = round(rouge_scores['rougeL'].fmeasure, 4)
        
        print(f"   -> Semántica: {semantic_score} | ROUGE-L: {rouge_l}")

        with open(archivo_reporte, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                model_name,
                training_time,
                final_loss,
                used_memory,
                inf_speed,
                semantic_score,
                rouge_l
            ])
            
        print(f"✅ {model_name} completado con éxito.")

    except Exception as e:
        print(f"ERROR CRÍTICO en {model_name}: {e}")
        with open(archivo_reporte, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([model_name, "ERROR", str(e), 0, 0, 0, 0])

    if 'model' in locals(): del model
    if 'tokenizer' in locals(): del tokenizer
    if 'trainer' in locals() and trainer is not None: del trainer
    limpiar_memoria()

print(f"\n PROCESO DE 3 ÉPOCAS TERMINADO. RESULTADOS EN: {archivo_reporte}")