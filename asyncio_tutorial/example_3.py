#Illustrates the correct usage of asyncio to run tasks concurrently
import asyncio
import time

async def fetch_data(param:int)->str:
    print(f"Do something with {param}...")
    await asyncio.sleep(param)
    print(f"Done with {param}")
    return f"Result of {param}"

async def main():
    # Now both the tasks are created and scheduled in the event loop.
    task1 = asyncio.create_task(fetch_data(1))
    task2 = asyncio.create_task(fetch_data(2)) 

    # Runs the task1 and suspends when it hits the asyncio.sleep() call, allowing task2 to run concurrently.
    # Task1 will resume when task2 hits its asyncio.sleep() call.
    result1 = await task1
    print("Task 1 fully completed")

    result2 = await task2
    print("Task 2 fully completed")
    return [result1, result2]

if __name__ == "__main__":
    t1 = time.perf_counter()
    
    results = asyncio.run(main())
    print(results)

    t2 = time.perf_counter()
    print(f"Total time taken: {t2 - t1:.2f} seconds")