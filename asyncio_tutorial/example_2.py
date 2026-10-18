# Wrong way of running async functions
import asyncio
import time

async def fetch_data(param:int)->str:
    print(f"Do something with {param}...")
    await asyncio.sleep(param)
    print(f"Done with {param}")
    return f"Result of {param}"

async def main():
    task1 = fetch_data(1)
    task2 = fetch_data(2) # These are only createing coroutine objects, not running them yet

    result1 = await task1 # This is scheduling the first coroutine to run and waiting for it to complete
    print("Task 1 fully completed")

    result2 = await task2 # This is scheduling the second coroutine to run and waiting for it to complete
    print("Task 2 fully completed")
    return [result1, result2]

if __name__ == "__main__":
    t1 = time.perf_counter()
    
    results = asyncio.run(main())
    print(results)

    t2 = time.perf_counter()
    print(f"Total time taken: {t2 - t1:.2f} seconds")