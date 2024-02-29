from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#model_path = './models/llama-2-13b-chat.Q4_K_M.gguf'
#model_path = './models/ggml-model-Q4_K_M.gguf'

class Loadllm:
    @staticmethod
    def load_llm(model):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        model_path = './models/' + model
        # Prepare the LLM
        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=40,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            callback_manager=callback_manager,
            verbose=True,
        )

        return llm
