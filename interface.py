
from header import *
from model import ModelFrame, qwen

# set this to True to print verbose output
DEBUG = ROOT_DEBUG or False

class SpellbookInterface:
	def __init__( self, retriever = None ):

		# How data is stored and retrieved from the local directory
		self.retriever = retriever

		# Message Logs
		self.messages = []

		# Model Settings
		self.thinking = False

		# Interface chat model

		#self.model_frame = gemma
		self.model_frame = qwen
		#self.model_frame = beluga

		# Load all the associated structures
		self.model, self.tokenizer, self.streamer = self.model_frame.get_model_set()


	# Calls an agent directly and collects/displays the response.
	def generate( self, prompt ):

		# Tokenize the text input
		inputs = self.tokenizer(
				[prompt],
				return_tensors = "pt"
			).to( self.model.device )

		# Send the prompt tokens to the model
		# and stream the response to the output
		response_data = self.model.generate(
			**inputs,
			streamer = self.streamer,
			max_new_tokens = 32768 # float('inf')
		)

		# Separate thinking tokens from the response.

		# Parse the resulting tokens into text
		response_tokens = response_data[0][len(inputs.input_ids[0]):].tolist() 
		completed_prompt = self.tokenizer.decode(
			response_tokens,
			skip_prompt = True,
			skip_special_tokens = True
		)
		return completed_prompt

	# Pings the user and adds resulting question to the message log
	def prompt_query( self ):

		# Display user name and wait for a question
		query = input(f"{self.model_frame.user}:\n")

		# Once the question comes, add it to the message list
		self.messages.append( { 'role': self.model_frame.user, 'content': query } )

		return


	# Format a message log into text for tokenizing.
	def format_messages( self, messages ):
		return self.tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True,
				enable_thinking=self.thinking # Switches between thinking and non-thinking modes. Default is True.
			)


	# Connect the user and an agent through the terminal output.
	def chat( self ):

		while True:

			# Formatting
			self.prompt_query()
			final_prompt = self.format_messages( self.messages )

			# Generate response
			response = self.generate( final_prompt )


if __name__ == '__main__':
	interface = SpellbookInterface()
	interface.chat()