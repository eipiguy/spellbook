import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig

from os import path

from loader import *

THIS_DIR = path.dirname( path.realpath( __file__ ) )
ASSET_DIR = path.join(THIS_DIR, 'assets')
MODEL_DIR = path.join(THIS_DIR, 'models')
EMBED_DIR = path.join(THIS_DIR, 'embeds')

class SpellbookInterface:

	def format_prompt( self, author, prompt ):
		return self.prompt_template.format( author = author, content = prompt ) + self.prompt_delim


	def format_qa( self, qa_list ):
		qa_prompt=""
		for qa in qa_list:
			qa_prompt += self.format_prompt( self.user_title, qa[0] )
			qa_prompt += self.format_prompt( self.llm_title, qa[1] )
		return qa_prompt


	def format_retrieval( self, raw_data ):
		retrieval_prompt = "The following information was retrieved in association with the given query. It may be useful, or it may be extraneous. Refer back to it only if it is prudent.\n\n"

		return self.format_prompt( self.self_title, retrieval_prompt + raw_data )


	def __init__( self, retriever ):

		self.retriever = retriever

		# Interface Chat Model Setup

		#self.chat_model_name = "Open-Orca/OpenOrca-Platypus2-13B"
		self.chat_model_name = "stabilityai/StableBeluga-7B"

		self.prompt_template = '\n### {author}:\n{content}'
		self.user_title = 'User'
		self.llm_title = 'Assistant'
		self.self_title = 'System'
		self.prompt_delim = '\n\n'

		quantization_config = BitsAndBytesConfig(
			llm_int8_enable_fp32_cpu_offload = True,
			load_in_8bit = True,
			bnb_8bit_quant_type = "nf8",
			bnb_8bit_use_double_quant = True,
			bnb_8bit_compute_dtype = torch.bfloat16
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

		memory = []
		data = []
		streamer = TextStreamer(
			self.chat_tokenizer,
			skip_prompt = True,
			skip_special_tokens = True
		)

		while True:
			query = input(f"### {self.user_title}:\n")

			# Retrieve any new associated data from the direct query.
			raw_relevant = self.retriever.retrieve(query)
			data.append( raw_relevant )

			# Reflect:
			# - Context
			# - Capabilities
			# - Desires
			# - Current Strategies

			# Self-Critique:
			# - Strengths/Weaknesses
			# - Play to strengths, strengthen weaknesses
			# - Most things take a lot of small, tedious steps. When learning, lean into your own aversions.

			# Plan (Resource Management):
			# - Progress towards strategies
			# - Return on investment graph
			# - Cost/benefit for each
			# - Sort and pursue

			# Formatting
			system_prompt = self.format_retrieval(data[-1])

			qa_prompt = self.format_qa(memory)
			final_prompt = system_prompt + qa_prompt + \
				self.format_prompt( self.user_title, query ) + \
				f"### {self.llm_title}:\n"

			# Generate response
			inputs = self.chat_tokenizer(
				final_prompt,
				return_tensors = "pt"
			).to( self.chat_model.device )

			# Info presented to user
			print( system_prompt )
			print(f"\n### {self.llm_title}:")


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

			# Add the question/answer into memory
			memory.append(
				[
					query,
					completed_prompt.split(f"{self.llm_title}:\n")[-1]
				]
			)


if __name__ == '__main__':
	interface = SpellbookInterface()
	interface.chat()