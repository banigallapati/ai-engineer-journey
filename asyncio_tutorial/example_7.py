# In this example, we will illustrate creating tasks in one shot and use gather to run them concurrently.
import asyncio
import time

async def fetch_data(param:int)->str:
    await asyncio.sleep(param)
    return f"Result of {param}"

async def main():
    # Create tasks manually i.e. one by one.
    task1 = asyncio.create_task(fetch_data(1))
    task2 = asyncio.create_task(fetch_data(2))
    result1 = await task1
    result2 = await task2
    print(f"Task 1 and 2 awaited results:{[result1, result2]}")

    # Create few coroutines simultaneously and run them concurrently using asyncio.gather().
    # Creating coroutines will not schedule them to run, we need to use asyncio.gather() to schedule.
    coroutines = [fetch_data(i) for i in range(1, 3)]
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    print(f"Coroutine Results:{results}")

    # Gather Tasks.
    # Here whenever tasks are created, they are automatically scheduled to run in the event loop.
    tasks = [asyncio.create_task(fetch_data(i)) for i in range(1, 3)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"Task Results:{results}")

    # Task Group. Use this when you want everything to be fail or everything to suceed.
    async with asyncio.TaskGroup() as tg:
        results = [tg.create_task(fetch_data(i)) for i in range(1, 3)]
        # All tasks are awaited when the context manager exits.

    print(f"Task Group Results:{[task.result() for task in results]}")
    return "Main Coroutine Done"

if __name__ == "__main__":
    t1 = time.perf_counter()
    
    results = asyncio.run(main())
    print(results)

    t2 = time.perf_counter()
    print(f"Total time taken: {t2 - t1:.2f} seconds")