import time

def measure_overhead():
    start = time.perf_counter()
    for _ in range(1000000):  # 1백만 번 호출
        time.perf_counter()
    end = time.perf_counter()
    return (end - start) / 1000000  # 평균 호출 시간

average_overhead = measure_overhead()
print(f"Average time.time() overhead: {average_overhead:.9f}초")