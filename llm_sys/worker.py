# llm sys must be imported before torch
# use this version of vllm for execution engine
# pip install git+https://github.com/vllm-project/vllm.git@v0.4.0.post1
import time
import llm_worker
import torch

from vllm.config import SchedulerConfig, CacheConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import Sequence, SequenceData, SequenceGroup, SequenceGroupMetadata, SequenceStatus
from vllm import SamplingParams

from llm_sys.engine.common import PipelineSequence, PipelineSequenceData, PipelineStageOut
from llm_sys.engine.scheduler import LayerwiseScheduler
from llm_sys.engine.exec_engine import PipelineStageEngine
import llm_sys.engine.llama

import llm_sys.utils as utils

from llm_sys.profiler.profiler import SlaveProfiler
import os

def init_engine(layer_ids, model_name, vram_usage=0.8):
    engine_args = EngineArgs(model=model_name, block_size=16,
                             load_format="dummy", enforce_eager=True,
                             swap_space=32, max_num_batched_tokens=4096,
                             gpu_memory_utilization=vram_usage)

    engine = PipelineStageEngine.from_engine_args(engine_args, layer_ids)
    return engine


def run_and_submit(engine, start_idx, end_idx, is_last_layer, hidden_size, force_decode) -> bool:
    """
    :return: whether parsed prompt in this iter
    """
    # Step 2.2: run inference layer by layer
    finished_seq_infos, output = [], None
    for layer_id in range(start_idx, end_idx):
        output = engine.step(layer_id=layer_id, finished_seq_infos=finished_seq_infos,
                             force_decode=force_decode)

    # Step 2.3: Prepare output
    if output == (None, None, None):
        # nothing to schedule, no need to re-enter
        return False, None
    if not is_last_layer:
        # output infos: list[((req_id (str), layer_id (int)), begin_idx (int), end_idx (int))]
        # output_tensor: a large tensor
        output_info_list, output_tensor, parsed_prompt = output
        output_tensor = output_tensor.cpu()

        # build finished reqeust information
        finished_ids, finished_offsets, finished_lengths = [], [], []
        # non-last, send out activations
        for output_info in output_info_list:
            finished_ids.append(int(output_info[0][0]))
            finished_offsets.append(output_info[1] * hidden_size)
            finished_lengths.append((output_info[2] - output_info[1]) * hidden_size)
    else:
        # last, send out int tensors (generated token ids)
        finished_ids, finished_offsets, finished_lengths = [], [], []
        generated_tokens = []
        parsed_prompt = output[1]
        for request_output in output[0]:
            cur_request_id = int(request_output.request_id[0])
            finished_ids.append(cur_request_id)
            finished_offsets.append(len(generated_tokens))
            finished_lengths.append(1)
            generated_tokens.append(1)
        output_tensor = torch.tensor(generated_tokens, dtype=torch.int32)

    # ------------------------------------------------------------------------------------------- #
    # Step 3: submit results
    # start_idx_list: List[int], offset in number of elements
    # length_list: List[int], length in number of elements
    # results_tensor: one large CPU tensor
    
    # Added by LJH
    time_stamp = time.time()
    llm_worker.submit_requests(finished_ids, finished_offsets, finished_lengths, output_tensor)
    # ------------------------------------------------------------------------------------------- #
    return parsed_prompt, time_stamp


def run_worker(scheduling_method: str, model_name: str, result_logging_dir: str, duration: int, vram_usage=0.8):
    # warm up gpu and initialize llm_sys
    utils.warm_up()
    worker_ip: str = utils.get_local_ip()
    print(f'worker_ip:{worker_ip}')
    # assert worker_ip.startswith("10"), "Local IP must start with 10"
    llm_worker.start_network_threads(utils.CONFIG_BROADCAST_ADDR, worker_ip, scheduling_method)
    start_idx, end_idx, is_last_layer = llm_worker.get_model_start_end_idx()
    print(f"[Python] Cluster initialization finished!")
    print(f"[Python] Model layers: [{start_idx}, {end_idx}).")
    print(f"[Python] Does this node output the last layer: {is_last_layer}.")
    
    # Added by LJH
    # ------------------------------------- Init Time ------------------------------------ #
    slave_profiler = SlaveProfiler(duration=120, ip_address=worker_ip, port=9001)
    slave_profiler.start_slave_profiling()
    # ------------------------------------------------------------------------------------ #

    # init vllm
    layer_ids = list(range(start_idx, end_idx))
    engine: PipelineStageEngine = init_engine(layer_ids, model_name, vram_usage=vram_usage)
    hidden_size = engine.model_config.get_hidden_size()

    last_log_time = time.time()
    
    # Added by LJH
    ground_zero = time.time()
    # time - query id - in/out - phase - context_len - this_iter_processed
    events = []
    
    while True:
        now = time.time() - ground_zero
        if now > slave_profiler.duration + 45:
            break
        
        # ------------------------------------------------------------------------------------------- #
        # Step 1: fetch new requests to compute on
        # request_ids: List[int]
        # is_prompt_list: List[bool]
        # start_layer_idx_list: List[int]
        # end_layer_idx_list: List[int]
        # num_tokens_list: List[int]
        #   1. for prompt, its the number of input tokens
        #   2. for decode, its the context size
        # max_tokens_list: List[int]
        #   1. in decode phase, if num_tokens (context) = max_tokens - 1, then we can release kv cache
        #   2. FIXME: be careful about the case when decode has length 0
        # offsets: List[int] (in number of elements)
        # lengths: List[int] (in number of elements)
        # is_token_tensor_list: List[bool]
        #   1. if true, offset and length belong to token tensor
        #   2. if false, offset and length belong to activation tensor
        # token_tensor: Torch.tensor on CUDA:0, torch.int32 (token ids)
        # activation_tensor: Torch.tensor on CUDA:0, torch.fp16 (activations)
        request_ids, is_prompt_list, start_layer_idx_list, end_layer_idx_list, num_tokens_list, max_tokens_list, \
            offsets, lengths, is_token_tensor_list, token_tensor, activation_tensor = llm_worker.fetch_new_requests()
        # ------------------------------------------------------------------------------------------- #
        time_stamp = time.time()
        
        # Step 2: run vllm
        # Step 2.1: put requests into the queues (register in vllm)
        for (request_id, is_prompt, start_layer_idx, end_layer_idx, num_tokens, max_tokens, offset,
             length, is_token) in zip(
            request_ids, is_prompt_list, start_layer_idx_list, end_layer_idx_list, num_tokens_list,
            max_tokens_list, offsets, lengths, is_token_tensor_list
        ):
            if is_prompt:
                # Added by LJH
                events.append((time_stamp, request_id, "in", "prompt", 0, num_tokens))
                
                print(f"[Prompt] Request {request_id} arrives (input_len={num_tokens}, max_len={max_tokens}, "
                      f"layers=[{start_layer_idx}, {end_layer_idx}))")
                # prompt phase requests
                if is_token:
                    # first layer: input tensor should be none
                    assert start_layer_idx == 0, "Only request that will infer layer 0 use token tensor!"
                    prompt_token_ids = list(token_tensor[offset: offset + length])
                    sampling_params = SamplingParams()
                    sampling_params.max_tokens = max_tokens - num_tokens  # it's generated tokens
                    sampling_params.ignore_eos = True
                    engine.add_request(f"{request_id}", "", sampling_params,
                                       local_layers=(start_layer_idx, end_layer_idx - 1),  # inclusive, thus -1
                                       seq_id=request_id, input_tensor=None,
                                       prompt_token_ids=prompt_token_ids)
                else:
                    # non-first layer: prompt token ids should be none
                    assert not start_layer_idx == 0, "Request that will infer layer 0 must use token tensor!"
                    input_tensor = activation_tensor[offset: offset + length].reshape(num_tokens, hidden_size)
                    sampling_params = SamplingParams()
                    sampling_params.max_tokens = max_tokens - num_tokens  # it's generated tokens
                    sampling_params.ignore_eos = True
                    engine.add_request(f"{request_id}", "", sampling_params,
                                       local_layers=(start_layer_idx, end_layer_idx - 1),  # inclusive, thus -1
                                       seq_id=request_id, input_tensor=input_tensor,
                                       prompt_token_ids=[1] * num_tokens)
            else:
                # Added by LJH
                events.append((time_stamp, request_id, "in", "decord", num_tokens, 1))
                
                # print(f"[Decode] Request {request_id} arrives (context_len={num_tokens}, max_len={max_tokens}, "
                #       f"layers=[{start_layer_idx}, {end_layer_idx}))")
                if is_token:
                    # first layer: no activations
                    engine.scheduler.update_req_data(start_layer_idx,
                                                     (f"{request_id}", start_layer_idx),
                                                     {(request_id, start_layer_idx): None})
                else:
                    input_tensor = activation_tensor[offset: offset + length].reshape(1, hidden_size)
                    engine.scheduler.update_req_data(start_layer_idx,
                                                     (f"{request_id}", start_layer_idx),
                                                     {(request_id, start_layer_idx): input_tensor})

        # step 2.2 & 2.3: run vllm and submit
        # Modified by LJH
        parsed_prompt, _ = run_and_submit(engine=engine, start_idx=start_idx, end_idx=end_idx, is_last_layer=is_last_layer,
                                       hidden_size=hidden_size, force_decode=False)
        # Added by LJH
        if time_stamp is not None:
            events.append((time_stamp, request_id, "out", "decord", "_", 1))
        if parsed_prompt:
            # Case for (prompt, out)
            parsed_prompt, time_stamp = run_and_submit(engine=engine, start_idx=start_idx, end_idx=end_idx,
                                           is_last_layer=is_last_layer, hidden_size=hidden_size,
                                           force_decode=True)
            # Added by LJH
            events.append((time_stamp, request_id, "out", "prompt", 0, "_"))
            assert not parsed_prompt, "Parsed prompt twice!"
        
        
    
        # log kv cache status
        if time.time() > last_log_time + 2:
            last_log_time = time.time()
            # GPU KV Cache Usage in %.
            num_total_gpu = engine.cache_config.num_gpu_blocks
            num_free_gpu = engine.scheduler.block_manager.get_num_free_gpu_blocks()
            gpu_cache_usage = 1.0 - (num_free_gpu / num_total_gpu)

            # CPU KV Cache Usage in %
            num_total_cpu = engine.cache_config.num_cpu_blocks
            cpu_cache_usage = 0.
            if num_total_cpu > 0:
                num_free_cpu = engine.scheduler.block_manager.get_num_free_cpu_blocks()
                cpu_cache_usage = 1.0 - (num_free_cpu / num_total_cpu)

            print(f"Time: {now} | "
                  f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}% - "
                  f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%.")
            if cpu_cache_usage > 0.1:
                print("Warning: CPU KV cache usage is larger than 0.1, considering lower arrival rate!")
    
    print("worker finished!")
    # Added by LJH
    # After break, save logging to files.
    events_file_name = os.path.join(result_logging_dir, "events.txt")
    with open(events_file_name, "w") as file:
        for item in events:
            file.write(f"{item}\n")
