import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig

from os import path

THIS_DIR = path.dirname( path.realpath( __file__ ) )
ASSET_DIR = path.join(THIS_DIR, 'assets')
MODEL_DIR = path.join(THIS_DIR, 'models')
EMBED_DIR = path.join(THIS_DIR, 'embeds')

class SpellbookInterface:	
	def template_chat(self, user, assistant = "", system = ""):
		system_template = ''
		if( system ):
			system_template = self.system_template.format(system=system)
		user_template = self.system_template.format(user=user)
		assistant_template = self.assistant_template.format(assistant=assistant)

		return system_template + user_template + assistant_template

	def __init__(self):
		
		#self.chat_model_name = "Open-Orca/OpenOrca-Platypus2-13B"
		self.chat_model_name = "stabilityai/StableBeluga-7B"

		self.prompt_template = """\
### {author}:
{content}"""
		self.prompt_delim = '\n\n'

		quantization_config = BitsAndBytesConfig(
			llm_int8_enable_fp32_cpu_offload = True,
			load_in_8bit=True,
			bnb_8bit_quant_type="nf8",
			bnb_8bit_use_double_quant=True,
			bnb_8bit_compute_dtype=torch.bfloat16
		)
		self.chat_tokenizer = AutoTokenizer.from_pretrained( self.chat_model_name, cache_dir = MODEL_DIR )
		self.chat_model = AutoModelForCausalLM.from_pretrained(
			self.chat_model_name,
			cache_dir = MODEL_DIR,
			offload_folder = MODEL_DIR,
			torch_dtype = torch.float16,
			low_cpu_mem_usage = True,
			device_map = "auto",
			quantization_config = quantization_config,
			rope_scaling = {"type": "dynamic", "factor": 2} # allows handling of longer inputs
		)

	def chat(self):
		user_title = 'User'
		llm_title = 'Assistant'
		memory = []
		streamer = TextStreamer(
			self.chat_tokenizer,
			skip_prompt = True,
			skip_special_tokens = True
		)

		while True:
			query = input(f"### {user_title}:\n")

			# Format recent memory data
			system_prompt = self.prompt_template.format( author = 'System', content= '' ) + self.prompt_delim
			memory_prompt=""
			for i,qa in enumerate(memory):
				memory_prompt += self.prompt_template.format( author = user_title, content = qa[0] ) + self.prompt_delim
				memory_prompt += self.prompt_template.format( author = llm_title, content = qa[1] ) + self.prompt_delim

			# Make the templated input for the LLM
			prompt = system_prompt + memory_prompt + self.prompt_template.format( author = user_title, content = query ) + self.prompt_delim + f"### {llm_title}:\n"

			inputs = self.chat_tokenizer( prompt, return_tensors = "pt" ).to( self.chat_model.device )

			print(f"\n### {llm_title}:")
			response = self.chat_model.generate(
				**inputs,
				streamer = streamer,
				use_cache = True,
				do_sample = True,
				top_p = 0.95,
				top_k = 0,
				max_new_tokens = 1024# float('inf')
			)
			completed_prompt = self.chat_tokenizer.decode(
				response[0],
				skip_prompt = True,
				skip_special_tokens = True
			)
			print()
			memory.append([query,completed_prompt.split(f"{llm_title}:\n")[-1]])


if __name__ == '__main__':
	interface = SpellbookInterface()
	interface.chat()