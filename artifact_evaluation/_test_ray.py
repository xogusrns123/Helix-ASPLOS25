import ray

ray.init()

def launch_script():
    import os
    os.system('bash /home/meiyixuan2000/helix/artifact_evaluation/launch_worker.sh')

refs = []
for node in ray.nodes():
    if 'GPU' in node['Resources'] and node['Resources']['GPU'] > 0:
        node_ip = 'node' + node['NodeName']
        # launch_remote_fn = ray.remote(resources={node_ip: 1})(launch_script)
        launch_remote_fn = ray.remote(num_gpus=1)(launch_script)
        ref = launch_remote_fn.remote()
        refs.append(ref)
    else:
        continue

ray.get(refs)

ray.shutdown()
