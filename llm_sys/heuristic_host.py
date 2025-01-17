# 2024.04.25 Yixuan Mei

# 2024.01.09 Lee JiHyuk
# The default code contains issues related to logging.
# The logs for entry and exit events are timestamped identically,
# making it impossible to distinguish the exact timing of in/out events on the host.

import llm_host
import os
import time

from simulator.trace_generator.trace_generator import TraceGenerator, ArrivalRateSource, Dataset, LengthSampler
from llm_sys.utils import get_local_ip, make_self_config, FlyingQuery, HOST_CONFIG_BROADCAST_ADDR

from llm_sys.profiler.profiler import get_device_ip_configs, MasterProfiler
from typing import Tuple, List

def run_heuristic_host_online(
        # scheduler
        scheduler_name: str,
        # cluster
        real_sys_config_file_name: str,
        # host device
        host_file_path: str, 
        # throughput
        avg_throughput: float,
        duration: int,
        # result
        result_logging_dir: str,
        device_num: int
) -> None:
    """
    Run host with !!![Swarm/Random + Online mode]!!!.
    """
    assert scheduler_name == "swarm" or scheduler_name == "random", "Scheduler must be either swarm or random!"
    print(f"Initializing host with {scheduler_name} scheduling!")

    # ------------------------------------- Online Generator ------------------------------------ #
    trace_generator = TraceGenerator(arrival_rate_source=ArrivalRateSource.AzureConv,
                                     length_dataset=Dataset.AzureConversation,
                                     cluster_token_throughput=avg_throughput, seed=0)
    trace = trace_generator.generate_trace(start_time=0, duration=duration)
    # ------------------------------------------------------------------------------------------- #

    # ------------------------------------- Init System ------------------------------------ #
    # make 'device_config.txt' and return worker_ip
    host_ip, open_port = make_self_config(device_num)
    print(f'host_ip:{host_ip}')
    
    global HOST_CONFIG_BROADCAST_ADDR
    HOST_CONFIG_BROADCAST_ADDR = "tcp://0.0.0.0:" + str(open_port)
    
    # assert host_ip.startswith("10"), "Local IP must be of form 10.xxx.xxx.xxx"
    llm_host.start_network_threads(HOST_CONFIG_BROADCAST_ADDR, host_ip, real_sys_config_file_name, 
                                   host_file_path, scheduler_name, device_num)
    time.sleep(20)
    print("[Python] Cluster initialization finished!")
    # -------------------------------------------------------------------------------------- #
    
    # Added by LJH
    # ------------------------------------- Init Profiler ------------------------------------ #
    # (compute_node_index, ip_address, open_port) 
    slave_configs: List[Tuple[int, str, int]] = get_device_ip_configs(real_sys_config_file_name)
    master_profiler = MasterProfiler(slave_configs, duration=duration, file_directory=result_logging_dir)
    master_profiler.start_master_profiling()
    # ------------------------------------------------------------------------------------ #
    
    once_check = True
    llm_host.signal_cluster_start(device_num)
    
    ground_zero = time.time()
    next_query_id = 0
    flying_queries_dict = {}
    # -----  log items ----- #
    # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
    query_routes = []
    # time - query id - in/out - phase - context_len - this_iter_processed
    events = []
    # ---------------------- #
    while True:
        # get time
        now = time.time() - ground_zero
        if now > master_profiler.duration + 30:
            break
        
        # For Adding New Request!
        # send new requests into cluster if needed
        while not len(trace) == 0 and trace[0][0] <= now:
            # the request has a time stamp smaller than now, should be sent
            expected_submit_time, input_length, output_length = trace.pop(0)

            # get query id
            cur_query_id = next_query_id
            next_query_id += 1

            # send it into the cluster (system will take care of routing)
            llm_host.launch_request(
                "prompt",  # request_type
                cur_query_id,  # request_id
                input_length,  # num_tokens
                input_length + output_length,  # max_num_tokens
                [i for i in range(input_length)],  # token_ids
                False,  # set_routing
                [],  # server_ids
                [],  # start_layer_ids
                [],  # end_layer_ids
            )

            # put into flying queries
            # time_stamp is Added by LJH
            # At the case of 'out', 'now' should be captured only BEFORE flying query being sent.
            time_stamp = time.time()
            flying_queries_dict[cur_query_id] = FlyingQuery(query_uid=cur_query_id,
                                                            input_length=input_length,
                                                            output_length=output_length,
                                                            compute_node_uids=None,
                                                            start_layers=None,
                                                            end_layers=None,
                                                            pipeline=None)
            
            # record_event is Modified by LJH
            # This code belongs to the control node, where
            # ('prompt', 'out') signifies the initiation of a query process.
            # 'out' and 'prompt' indicate the request signal to compute nodes for starting a prefill stage.
        # Original code
            # events.append((time_stamp, cur_query_id, "out", "prompt", 0, input_length + 1))
        # Modified code
            master_profiler.record_event(time_stamp, cur_query_id, "out", "prompt", 0, input_length + 1)
            print(f"Send out new query {cur_query_id}, input len = {input_length}, "
                  f"max_len = {input_length + output_length}")
        
        # get finished requests
        now = time.time() - ground_zero
        finished_query_ids, generated_token_ids, routes, num_layers = llm_host.gather_finished_requests()
        # Added by LJH
        # 'in' indicate the completion signal from compute nodes for each stage.
        # At the case of 'in', 'now' should be captured only AFTER all finished requests have been gathered.
        time_stamp = time.time()
        
        for query_uid, route_list, num_layer_list in zip(finished_query_ids, routes, num_layers):
            # first receive the message
            py_on_the_fly_query = flying_queries_dict[query_uid]
            
            if py_on_the_fly_query.processed_tokens == 0:
                # prompt phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                
                # Modified by LJH
            # Original code
                # events.append((time_stamp, query_uid, "in", "prompt", 0, py_on_the_fly_query.input_length + 1))
            # Modified code
                master_profiler.record_event(time_stamp, query_uid, "in", "prompt", 0, py_on_the_fly_query.input_length + 1)
                py_on_the_fly_query.processed_tokens += py_on_the_fly_query.input_length + 1

                # now we can log the request with its route
                # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
                cur_log_start = 0
                start_layer_ids, end_layer_ids = [], []
                for num_layer in num_layer_list:
                    start_layer_ids.append(cur_log_start)
                    end_layer_ids.append(cur_log_start + num_layer)
                    cur_log_start += num_layer
                query_routes.append((query_uid, py_on_the_fly_query.input_length, py_on_the_fly_query.output_length,
                                     route_list, start_layer_ids, end_layer_ids))
            else:
                # decode phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                
                # Modified by LJH
            # Original code
                # events.append((time_stamp, query_uid, "in", "decode", py_on_the_fly_query.processed_tokens, 1))
            # Modified code
                master_profiler.record_event(time_stamp, query_uid, "in", "decode", py_on_the_fly_query.processed_tokens, 1)
                
                py_on_the_fly_query.processed_tokens += 1

            # then we decide whether to send out new messages (decodes)
            max_size = py_on_the_fly_query.input_length + py_on_the_fly_query.output_length
            assert py_on_the_fly_query.processed_tokens <= max_size, "Found request that did not end!"
            if py_on_the_fly_query.processed_tokens == max_size:
                # not send: finished, remove from expectations
                del flying_queries_dict[query_uid]
                print(f"Query {query_uid}, finished (total_len={py_on_the_fly_query.processed_tokens})")

            else:
                # then we send the query back into the cluster
                
                # Added by LJH
                # 'out' and 'decode' indicate the request signal to compute nodes for starting a decode stage.
                # At the case of 'out', 'now' should be captured only BEFORE flying query being sent.
                time_stamp = time.time()
                llm_host.launch_request(
                    "decode",  # request_type
                    query_uid,  # request_id
                    py_on_the_fly_query.processed_tokens,  # num_tokens (context size)
                    max_size,  # max_num_tokens
                    [-1],  # token_ids
                    False,  # set_routing
                    [],  # server_ids
                    [],  # start_layer_ids
                    [],  # end_layer_ids
                )

                # time - query id - in/out - phase - context_len - this_iter_processed
                
                # Modified by LJH
            # Original code
                # events.append((time_stamp, query_uid, "out", "decode", py_on_the_fly_query.processed_tokens, 1))
            # Modified code
                master_profiler.record_event(time_stamp, query_uid, "out", "decode", py_on_the_fly_query.processed_tokens, 1)

    # save logging files
    print(f"Queries still flying: {flying_queries_dict.keys()}.")
    query_routes_file_name = os.path.join(result_logging_dir, "query_route.txt")
    events_file_name = os.path.join(result_logging_dir, "events.txt")
    with open(query_routes_file_name, "w") as f:
        for item in query_routes:
            f.write(f"{item}\n")
    # Modified by LJH
# Original code
    # with open(events_file_name, "w") as f:
    #     for item in events:
    #         f.write(f"{item}\n")
# Modified code
    master_profiler.write_event_to_csv()
    master_profiler.generate_delay_report()
    print("run_heuristic_host_online has been finished!")


def run_heuristic_host_offline(
        # scheduler
        scheduler_name: str,
        # cluster
        real_sys_config_file_name: str,
        # throughput
        initial_launch_num: int,
        duration: int,
        # result
        result_logging_dir: str,
) -> None:
    """
    Run host with !!![Swarm/Random + Offline mode]!!!.
    """
    assert scheduler_name == "swarm" or scheduler_name == "random", "Scheduler must be either swarm or random!"
    print(f"Initializing host with {scheduler_name} scheduling!")

    # ------------------------------------- Offline Initial ------------------------------------- #
    length_sampler = LengthSampler(dataset=Dataset.AzureConversation, seed=0)
    initial_requests = []
    for i in range(initial_launch_num):
        request_time = 0.1 + i * 0.1
        input_length, output_length = length_sampler.sample_length()
        initial_requests.append((request_time, input_length, output_length))
    # ------------------------------------------------------------------------------------------- #

    # ------------------------------------- Init System ------------------------------------ #
    host_ip: str = get_local_ip()
    print(f'host_ip:{host_ip}')
    # assert host_ip.startswith("10"), "Local IP must be of form 10.xxx.xxx.xxx"
    llm_host.start_network_threads(HOST_CONFIG_BROADCAST_ADDR, host_ip, real_sys_config_file_name, scheduler_name)
    time.sleep(20)
    print("[Python] Cluster initialization finished!")
    # -------------------------------------------------------------------------------------- #
    ground_zero = time.time()
    next_query_id = 0
    flying_queries_dict = {}
    # -----  log items ----- #
    # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
    query_routes = []
    # time - query id - in/out - phase - context_len - this_iter_processed
    events = []
    # ---------------------- #
    last_log_time = 0
    while True:
        # get time
        now = time.time() - ground_zero
        if now > duration + 1:
            break

        if now - last_log_time > 1:
            print(f"[t={now}]")
            last_log_time = now

        # send out initial requests into the cluster
        while not len(initial_requests) == 0 and initial_requests[0][0] <= now:
            # the request has a time stamp smaller than now, should be sent
            expected_submit_time, input_length, output_length = initial_requests.pop(0)

            # get query id
            cur_query_id = next_query_id
            next_query_id += 1

            # send it into the cluster
            llm_host.launch_request(
                "prompt",  # request_type
                cur_query_id,  # request_id
                input_length,  # num_tokens
                input_length + output_length,  # max_num_tokens
                [i for i in range(input_length)],  # token_ids
                False,  # set_routing
                [],  # server_ids
                [],  # start_layer_ids
                [],  # end_layer_ids
            )

            # put into flying queries
            flying_queries_dict[cur_query_id] = FlyingQuery(query_uid=cur_query_id,
                                                            input_length=input_length,
                                                            output_length=output_length,
                                                            compute_node_uids=None,
                                                            start_layers=None,
                                                            end_layers=None,
                                                            pipeline=None)

            # time - query id - in/out - phase - context_len - this_iter_processed
            events.append((now, cur_query_id, "out", "prompt", 0, input_length + 1))
            print(f"Send out new query {cur_query_id}, input len = {input_length}, "
                  f"max_len = {input_length + output_length}")

        # get finished requests
        now = time.time() - ground_zero
        finished_query_ids, generated_token_ids, routes, num_layers = llm_host.gather_finished_requests()
        for query_uid, route_list, num_layer_list in zip(finished_query_ids, routes, num_layers):
            # first receive the message
            py_on_the_fly_query = flying_queries_dict[query_uid]
            if py_on_the_fly_query.processed_tokens == 0:
                # prompt phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "in", "prompt", 0, py_on_the_fly_query.input_length + 1))
                py_on_the_fly_query.processed_tokens += py_on_the_fly_query.input_length + 1

                # now we can log the request with its route
                # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
                cur_log_start = 0
                start_layer_ids, end_layer_ids = [], []
                for num_layer in num_layer_list:
                    start_layer_ids.append(cur_log_start)
                    end_layer_ids.append(cur_log_start + num_layer)
                    cur_log_start += num_layer
                query_routes.append((query_uid, py_on_the_fly_query.input_length, py_on_the_fly_query.output_length,
                                     route_list, start_layer_ids, end_layer_ids))

            else:
                # decode phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "in", "decode", py_on_the_fly_query.processed_tokens, 1))
                py_on_the_fly_query.processed_tokens += 1

            # then we decide whether to send out new messages (decodes)
            max_size = py_on_the_fly_query.input_length + py_on_the_fly_query.output_length
            assert py_on_the_fly_query.processed_tokens <= max_size, "Found request that did not end!"
            if py_on_the_fly_query.processed_tokens == max_size:
                # not send: finished, remove from expectations
                del flying_queries_dict[query_uid]
                print(f"Query {query_uid}, finished (total_len={py_on_the_fly_query.processed_tokens})")

                # send a new query to replace the old one
                input_length, output_length = length_sampler.sample_length()

                # get query id
                cur_query_id = next_query_id
                next_query_id += 1

                # send it into the cluster
                llm_host.launch_request(
                    "prompt",  # request_type
                    cur_query_id,  # request_id
                    input_length,  # num_tokens
                    input_length + output_length,  # max_num_tokens
                    [i for i in range(input_length)],  # token_ids
                    False,  # set_routing
                    [],  # server_ids
                    [],  # start_layer_ids
                    [],  # end_layer_ids
                )

                # put into flying queries
                flying_queries_dict[cur_query_id] = FlyingQuery(query_uid=cur_query_id,
                                                                input_length=input_length,
                                                                output_length=output_length,
                                                                compute_node_uids=None,
                                                                start_layers=None,
                                                                end_layers=None,
                                                                pipeline=None)

                # save log
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, cur_query_id, "out", "prompt", 0, input_length + 1))
                print(f"Send out new query {cur_query_id}, input len = {input_length}, "
                      f"max_len = {input_length + output_length} (decode finish request replacement)")

            else:
                # then we send the query back into the cluster (next decode iter)
                llm_host.launch_request(
                    "decode",  # request_type
                    query_uid,  # request_id
                    py_on_the_fly_query.processed_tokens,  # num_tokens (context size)
                    max_size,  # max_num_tokens
                    [-1],  # token_ids
                    False,  # set_routing
                    [],  # server_ids
                    [],  # start_layer_ids
                    [],  # end_layer_ids
                )

                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "out", "decode", py_on_the_fly_query.processed_tokens, 1))

    # save logging files
    print(f"Queries still flying: {flying_queries_dict.keys()}.")
    query_routes_file_name = os.path.join(result_logging_dir, "query_route.txt")
    events_file_name = os.path.join(result_logging_dir, "events.txt")
    with open(query_routes_file_name, "w") as f:
        for item in query_routes:
            f.write(f"{item}\n")
    with open(events_file_name, "w") as f:
        for item in events:
            f.write(f"{item}\n")


def run_heuristic_host_profiling(
        duration: int,
        # batch
        initial_launch_num: int,
        seq_len: int,
        output_len: int,
        device_num: int
) -> None:
    """
    Run host with !!![Swarm/Random + Offline mode]!!!.
    """
    # scheduler
    scheduler_name: str = "random"
    
    # make directory
    experiment_name: str = f"batch_{initial_launch_num}_device_num_{device_num}"
    result_logging_dir: str = f"./profiling/{experiment_name}"
    os.makedirs(result_logging_dir, exist_ok=True)
    
    # cluster config file path
    real_sys_config_file_name: str = f"{result_logging_dir}/real_sys_config.txt"
    # host device config file path
    host_file_path: str = f"./profiling/device_config.txt"
    
    print(f"Initializing host with {scheduler_name} scheduling!")
    # ------------------------------------- Offline Initial ------------------------------------- #
    length_sampler = LengthSampler(dataset=Dataset.AzureConversation, seed=0)
    initial_requests = []
    for i in range(initial_launch_num):
        request_time = 0.1 + i * 0.01
        input_length, output_length = length_sampler.sample_length()
        initial_requests.append((request_time, input_length, output_length))
    # ------------------------------------------------------------------------------------------- #

    # ------------------------------------- Init System ------------------------------------ #
    # make 'device_config.txt' and return worker_ip
    host_ip, open_port = make_self_config(device_num)
    print(f'host_ip:{host_ip}')
    
    global HOST_CONFIG_BROADCAST_ADDR
    HOST_CONFIG_BROADCAST_ADDR = "tcp://0.0.0.0:" + str(open_port)
    
    # assert host_ip.startswith("10"), "Local IP must be of form 10.xxx.xxx.xxx"
    llm_host.start_network_threads(HOST_CONFIG_BROADCAST_ADDR, host_ip, real_sys_config_file_name, 
                                   host_file_path, scheduler_name, device_num)
    time.sleep(20)
    print("[Python] Cluster initialization finished!")
    # -------------------------------------------------------------------------------------- #
    
    # Added by LJH
    # ------------------------------------- Init Profiler ------------------------------------ #
    # (compute_node_index, ip_address, open_port) 
    slave_configs: List[Tuple[int, str, int]] = get_device_ip_configs(real_sys_config_file_name)
    master_profiler = MasterProfiler(slave_configs, duration=duration, file_directory=result_logging_dir)
    master_profiler.start_master_profiling()
    # ------------------------------------------------------------------------------------ #
    
    # Make work nodes to start together
    llm_host.signal_cluster_start(device_num)
    
    ground_zero = time.time()
    next_query_id = 0
    flying_queries_dict = {}
    # -----  log items ----- #
    # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
    query_routes = []
    # time - query id - in/out - phase - context_len - this_iter_processed
    events = []
    # ---------------------- #
    last_log_time = 0
    while True:
        # get time
        now = time.time() - ground_zero
        if now > duration + 30:
            break

        if now - last_log_time > 5:
            print(f"[t={now}]")
            last_log_time = now

        # send out initial requests into the cluster
        while not len(initial_requests) == 0 and initial_requests[0][0] <= now:
            # the request has a time stamp smaller than now, should be sent
            expected_submit_time, input_length, output_length = initial_requests.pop(0)

            # get query id
            cur_query_id = next_query_id
            next_query_id += 1

            # send it into the cluster
            llm_host.launch_request(
                "prompt",  # request_type
                cur_query_id,  # request_id
                input_length,  # num_tokens
                input_length + output_length,  # max_num_tokens
                [i for i in range(input_length)],  # token_ids
                False,  # set_routing
                [],  # server_ids
                [],  # start_layer_ids
                [],  # end_layer_ids
            )

            # put into flying queries
            time_stamp = time.time()
            flying_queries_dict[cur_query_id] = FlyingQuery(query_uid=cur_query_id,
                                                            input_length=input_length,
                                                            output_length=output_length,
                                                            compute_node_uids=None,
                                                            start_layers=None,
                                                            end_layers=None,
                                                            pipeline=None)

            # time - query id - in/out - phase - context_len - this_iter_processed
            events.append((now, cur_query_id, "out", "prompt", 0, input_length + 1))
            master_profiler.record_event(time_stamp, cur_query_id, "out", "prompt", 0, input_length + 1)
            print(f"Send out new query {cur_query_id}, input len = {input_length}, "
                  f"max_len = {input_length + output_length}")

        # get finished requests
        now = time.time() - ground_zero
        time_stamp = time.time()
        finished_query_ids, generated_token_ids, routes, num_layers = llm_host.gather_finished_requests()
        for query_uid, route_list, num_layer_list in zip(finished_query_ids, routes, num_layers):
            # first receive the message
            py_on_the_fly_query = flying_queries_dict[query_uid]
            if py_on_the_fly_query.processed_tokens == 0:
                # prompt phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "in", "prompt", 0, py_on_the_fly_query.input_length + 1))
                master_profiler.record_event(time_stamp, query_uid, "in", "prompt", 0, py_on_the_fly_query.input_length + 1)
                py_on_the_fly_query.processed_tokens += py_on_the_fly_query.input_length + 1

                # now we can log the request with its route
                # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
                cur_log_start = 0
                start_layer_ids, end_layer_ids = [], []
                for num_layer in num_layer_list:
                    start_layer_ids.append(cur_log_start)
                    end_layer_ids.append(cur_log_start + num_layer)
                    cur_log_start += num_layer
                query_routes.append((query_uid, py_on_the_fly_query.input_length, py_on_the_fly_query.output_length,
                                     route_list, start_layer_ids, end_layer_ids))

            else:
                # decode phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "in", "decode", py_on_the_fly_query.processed_tokens, 1))
                master_profiler.record_event(time_stamp, query_uid, "in", "decode", py_on_the_fly_query.processed_tokens, 1)
                py_on_the_fly_query.processed_tokens += 1

            # then we decide whether to send out new messages (decodes)
            max_size = py_on_the_fly_query.input_length + py_on_the_fly_query.output_length
            assert py_on_the_fly_query.processed_tokens <= max_size, "Found request that did not end!"
            if py_on_the_fly_query.processed_tokens == max_size:
                # not send: finished, remove from expectations
                del flying_queries_dict[query_uid]
                print(f"Query {query_uid}, finished (total_len={py_on_the_fly_query.processed_tokens})")

                # send a new query to replace the old one
                input_length, output_length = length_sampler.sample_length()

                # get query id
                cur_query_id = next_query_id
                next_query_id += 1

                # send it into the cluster
                time_stamp = time.time()
                llm_host.launch_request(
                    "prompt",  # request_type
                    cur_query_id,  # request_id
                    input_length,  # num_tokens
                    input_length + output_length,  # max_num_tokens
                    [i for i in range(input_length)],  # token_ids
                    False,  # set_routing
                    [],  # server_ids
                    [],  # start_layer_ids
                    [],  # end_layer_ids
                )

                # put into flying queries
                flying_queries_dict[cur_query_id] = FlyingQuery(query_uid=cur_query_id,
                                                                input_length=input_length,
                                                                output_length=output_length,
                                                                compute_node_uids=None,
                                                                start_layers=None,
                                                                end_layers=None,
                                                                pipeline=None)

                # save log
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, cur_query_id, "out", "prompt", 0, input_length + 1))
                master_profiler.record_event(time_stamp, cur_query_id, "out", "prompt", py_on_the_fly_query.processed_tokens, 1)
                print(f"Send out new query {cur_query_id}, input len = {input_length}, "
                      f"max_len = {input_length + output_length} (decode finish request replacement)")

            else:
                # then we send the query back into the cluster (next decode iter)
                time_stamp = time.time()
                llm_host.launch_request(
                    "decode",  # request_type
                    query_uid,  # request_id
                    py_on_the_fly_query.processed_tokens,  # num_tokens (context size)
                    max_size,  # max_num_tokens
                    [-1],  # token_ids
                    False,  # set_routing
                    [],  # server_ids
                    [],  # start_layer_ids
                    [],  # end_layer_ids
                )

                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "out", "decode", py_on_the_fly_query.processed_tokens, 1))
                master_profiler.record_event(time_stamp, query_uid, "out", "decode", py_on_the_fly_query.processed_tokens, 1)

    # save logging files
    print(f"Queries still flying: {flying_queries_dict.keys()}.")
    query_routes_file_name = os.path.join(result_logging_dir, "query_route.txt")
    events_file_name = os.path.join(result_logging_dir, "events.txt")
    with open(query_routes_file_name, "w") as f:
        for item in query_routes:
            f.write(f"{item}\n")
    with open(events_file_name, "w") as f:
        for item in events:
            f.write(f"{item}\n")
    master_profiler.write_event_to_csv()
    master_profiler.generate_delay_report()
    print("run_heuristic_host_profiling has been finished!")
