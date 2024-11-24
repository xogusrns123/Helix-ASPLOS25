echo "Running simulation for helix llama30b offline"
python step3_simulation.py helix llama30b offline
sleep 5

echo "Running simulation for swarm llama30b offline"
python step3_simulation.py swarm llama30b offline
sleep 5

echo "Running simulation for separate pipelines llama30b offline"
python step3_simulation.py separate llama30b offline
sleep 5

echo "Running simulation for helix llama30b online"
python step3_simulation.py helix llama30b online
sleep 5

echo "Running simulation for swarm llama30b online"
python step3_simulation.py swarm llama30b online
sleep 5
echo "Running simulation for separate pipelines llama30b online"
python step3_simulation.py separate llama30b online
sleep 5

echo "Running simulation for helix llama70b offline"
python step3_simulation.py helix llama70b offline
sleep 5

echo "Running simulation for swarm llama70b offline"
python step3_simulation.py swarm llama70b offline
sleep 5

echo "Running simulation for separate pipelines llama70b offline"
python step3_simulation.py separate llama70b offline
sleep 5

echo "Running simulation for helix llama70b online"
python step3_simulation.py helix llama70b online
sleep 5

echo "Running simulation for swarm llama70b online"
python step3_simulation.py swarm llama70b online
sleep 5

echo "Running simulation for separate pipelines llama70b online"
python step3_simulation.py separate llama70b online