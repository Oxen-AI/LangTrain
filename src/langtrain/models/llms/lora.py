from typing import List

from peft import LoraConfig, get_peft_model, PeftModel
import torch
import transformers

def load_model(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # target_modules: List[str] = [
    #     "c_attn",
    # ],
    target_modules=["q", "v"]
) -> PeftModel:
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        fan_in_fan_out=True,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def load_model_inference(
    base_model,
    checkpoint_name,
    device: str = "cuda"
) -> PeftModel:
    print(f"Loading checkpoint: {checkpoint_name}")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = PeftModel.from_pretrained(
        base_model, checkpoint_name, torch_dtype=dtype
    )
    model.half()
    model.eval()
    
    if torch.__version__ >= "2":
        model = torch.compile(model)
    
    return model
    
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     checkpoint_name,
    #     load_in_8bit=True,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )
    # return model


def inference(
    llm,
    prompt: str,
) -> str:
    tokenizer = llm.tokenizer
    model = llm.model
    # model.to(model.device)
    model.eval()
    tokens = tokenizer(prompt, padding=False, return_tensors="pt")# .to(model.device)
    input_ids = tokens["input_ids"]
    attention_masks = tokens["attention_mask"]

    print("tokenizer.pad_token_id")
    print(tokenizer.pad_token_id)
    
    print("tokenizer.eos_token_id")
    print(tokenizer.eos_token_id)


    generation_config = transformers.GenerationConfig(
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.75,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=False,
        early_stopping=True,
        # num_beams=5,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
    )

    print("input_ids")
    print(input_ids)
    print("attention_masks")
    print(attention_masks)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        generation_config=generation_config
    )
    print("output_ids")
    print(output_ids)
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    print("result")
    print(result)

    return llm.decode_output(tokenizer, input_ids[0], output_ids[0]).strip()