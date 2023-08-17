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
			system_template = f"### System:\n{system}\n\n"
		user_template = f"### User:\n{user}\n\n"
		assistant_template = f"### Assistant:\n{assistant}"

		return system_template + user_template + assistant_template
	
	def __init__(self):
		
		#self.chat_model_name = "Open-Orca/OpenOrca-Platypus2-13B"
		self.chat_model_name = "stabilityai/StableBeluga-7B"

		quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
		self.chat_tokenizer = AutoTokenizer.from_pretrained( self.chat_model_name, cache_dir = MODEL_DIR )
		self.chat_model = AutoModelForCausalLM.from_pretrained(
			self.chat_model_name,
			cache_dir = MODEL_DIR,
			offload_folder = MODEL_DIR,
			torch_dtype = torch.float16,
			low_cpu_mem_usage = True,
			device_map = "auto",
			load_in_8bit = True,
			quantization_config = quantization_config,
			rope_scaling = {"type": "dynamic", "factor": 2} # allows handling of longer inputs
		)

	def chat(self):

		while True:
			prompt = input("User:\n")
			templated_prompt = self.template_chat(prompt)

			inputs = self.chat_tokenizer( templated_prompt, return_tensors = "pt" ).to( self.chat_model.device )
			#del inputs['token_type_ids']

			streamer = TextStreamer(
				self.chat_tokenizer,
				skip_prompt = True,
				skip_special_tokens = True
			)

			output = self.chat_model.generate(
				**inputs,
				streamer = streamer,
				use_cache = True,
				do_sample=True,
				top_p=0.95,
				top_k=0,
				max_new_tokens = 256# float('inf')
			)
			output_text = self.chat_tokenizer.decode(
				output[0],
				skip_special_tokens = True
			)
			print(output_text)


if __name__ == '__main__':
	interface = SpellbookInterface()
	interface.chat()