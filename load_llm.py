import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    AutoModelForSequenceClassification
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import pandas as pd



from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM


# CODE TO LOAD THE FINETUNED LLM MODEL

config = PeftConfig.from_pretrained("LLAMA27B_XML_FINETUNED_MODEL")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(model, "LLAMA27B_XML_FINETUNED_MODEL")
tokenizer = AutoTokenizer.from_pretrained("LLAMA27B_XML_FINETUNED_MODEL")



# CODE TO GENERATE OUTPUT (XML SCRIPTS)

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline
prompt = "Generate me a new XML."
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
generated_text = result[0]['generated_text']


# Assuming result[0]['generated_text'] contains the generated text including [/INST] markers
generated_text = result[0]['generated_text']

# Find the position of the first [/INST] marker and add its length to start extracting after it
start_pos = generated_text.find('[/INST]') + len('[/INST]')

# Extract everything after the first [/INST] to the end of the string
content_after_first_inst = generated_text[start_pos:]

# Now, find the position of the next [/INST] in the extracted content, if it exists
end_pos = content_after_first_inst.find('[/INST]')

# If there's another [/INST], extract up to it; otherwise, use the whole content
if end_pos != -1:
    content = content_after_first_inst[:end_pos].strip()  # Use strip() to remove leading/trailing whitespace
else:
    content = content_after_first_inst.strip()  # Use strip() to remove leading/trailing whitespace

end_index = content.find('</processDiagram>') + len('</processDiagram>')

content = content[:end_index]



print(content)
