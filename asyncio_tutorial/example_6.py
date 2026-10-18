# In this example we illustrate, how we can concurrency by running in Threads and Processes.
# This will be useful when we have to run blocking synchronous code in our async code. 
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor

# Now we have a blocking synchronous function that is not awaitable.
def fetch_data(param:int)->str:
    print(f"Do something with {param}...", flush=True)
    time.sleep(param)
    print(f"Done with {param}", flush=True)
    return f"Result of {param}"

async def main():
    # Run in Threads
    task1 = asyncio.create_task(asyncio.to_thread(fetch_data, 1))
    task2 = asyncio.create_task(asyncio.to_thread(fetch_data, 2))

    result1 = await task1
    print("Thread 1 fully completed")

    result2 = await task2
    print("Thread 2 fully completed")

    # Run in Process Pool
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        task1 = loop.run_in_executor(executor, fetch_data, 1)
        task2 = loop.run_in_executor(executor, fetch_data, 2)

        result1 = await task1
        print("Process 1 fully completed")

        result2 = await task2
        print("Process 2 fully completed")

    return [result1, result2]

if __name__ == "__main__":
    t1 = time.perf_counter()
    
    results = asyncio.run(main())
    print(results)

    t2 = time.perf_counter()
    print(f"Total time taken: {t2 - t1:.2f} seconds")