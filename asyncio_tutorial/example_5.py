# Same as example 3, illustrates how synchronouse blocking calls can lose the benefits of concurrency.
import asyncio
import time

async def fetch_data(param:int)->str:
    print(f"Do something with {param}...")

    # This synchronous blocking call will block the event loop and prevent other tasks running concurrently.
    time.sleep(param)
    # await asyncio.sleep(param) # This is the correct way to do it, but we are using time.sleep().
    print(f"Done with {param}")
    return f"Result of {param}"

async def main():
    # Now both the tasks are created and scheduled in the event loop.
    task1 = asyncio.create_task(fetch_data(1))
    task2 = asyncio.create_task(fetch_data(2)) 

    # Waits till the  first task is completely run because of synchronous blocking call in the first task.
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