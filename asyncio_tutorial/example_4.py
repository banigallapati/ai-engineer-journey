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

    # As soon as the await is hit, control is returned to the event loop, allowing other tasks to run.
    # Usually FIFO (First In First Out) is used to schedule the tasks, but it is not guaranteed. 
    # The event loop can choose to run any of the scheduled tasks.
    # task1 can run first till it hits the await asyncio.sleep() call, 
    # then task2 can run till it hits its await asyncio.sleep() call, and then task1 can resume and complete
    # followed by task2 resuming and completing.
    # main event loop will wait till task2 is completed.
    result2 = await task2
    print("Task 2 fully completed")

    result1 = await task1
    print("Task 1 fully completed")
    return [result1, result2]

if __name__ == "__main__":
    t1 = time.perf_counter()
    
    results = asyncio.run(main())
    print(results)

    t2 = time.perf_counter()
    print(f"Total time taken: {t2 - t1:.2f} seconds")