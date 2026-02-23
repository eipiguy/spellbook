from torch import float16, bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

from header import *

# set this to True to print verbose output
DEBUG = ROOT_DEBUG or False

quant_4bit = BitsAndBytesConfig(
	llm_int4_enable_fp32_cpu_offload = True, # Use this is GPU RAM is too low.
	load_in_4bit = True,
	bnb_4bit_quant_type = "nf4",
	bnb_4bit_use_double_quant = True,
	bnb_4bit_compute_dtype = bfloat16
)

quant_8bit = BitsAndBytesConfig(
	llm_int8_enable_fp32_cpu_offload = True, # Use this is GPU RAM is too low.
	load_in_8bit = True,
	bnb_8bit_quant_type = "nf8",
	bnb_8bit_use_double_quant = True,
	bnb_8bit_compute_dtype = bfloat16
)

class ModelFrame:
	def __init__( self, name,
		quantization = quant_4bit,
		system = 'system',
		assistant = 'assistant',
		user = 'user',
		delimeter = '\n\n' ):

		# Model Specs
		self.name = name
		self.quantization = quantization

		# Model Prompt Format
		self.system =  system
		self.user = user
		self.assistant = assistant
		self.delimeter = delimeter

	def get_model_set( self ):

		model = AutoModelForCausalLM.from_pretrained(
			self.name,
			#attn_implementation="flash_attention_2",
			cache_dir = MODEL_DIR,
			offload_folder = MODEL_DIR,
			dtype = 'auto',
			low_cpu_mem_usage = True,
			device_map = 'auto',
			#quantization_config = self.quantization
		)

		tokenizer = AutoTokenizer.from_pretrained( self.name, cache_dir = MODEL_DIR )

		streamer = TextStreamer(
			tokenizer,
			skip_prompt = True,
			skip_special_tokens = True
		)
		return [ model, tokenizer, streamer ]

# 7B Models
beluga = ModelFrame( 'stabilityai/StableBeluga-7B' )

# 4B Models
qwen = ModelFrame( 'Qwen/Qwen3-4B' )
gemma = ModelFrame( 'google/gemma-3-4b-pt' )