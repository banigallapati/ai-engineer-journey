# Code in this folder are from the Youtube tutorial: https://www.youtube.com/watch?v=oAkLSJNr5zY
# Title of the video is "Python Tutorial: AsyncIO - Complete Guide to Asynchronous Programming with Animations"
import asyncio
import time

def sync_function(test_param: str)-> str:
    print("This is a synchronous function.")
    time.sleep(1)
    return f"Sync Result called with {test_param}"

# Also know as a coroutine function
async def async_function(test_param: str)-> str:
    print("This is an asynchronous function.")
    await asyncio.sleep(1)
    return f"Async Result called with {test_param}"

async def main():
    sync_result = sync_function("Sync Test")
    print(sync_result)

   # Futures are a way to represent a result that may not be available yet. They are used in asynchronous programming to handle operations that take time to complete, such as I/O operations or network requests. In Python's asyncio library, futures are represented by the `asyncio.Future` class.
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    print(f"Empty Future: {future}")

    future.set_result("Future Result: Test")
    future_result = await future
    print(future_result)

    coroutine_obj = async_function("Async Test")
    print(f"Coroutine Object: {coroutine_obj}")

    coroutine_result = await coroutine_obj
    print(coroutine_result)

    task = asyncio.create_task(async_function("Async Test with Task"))
    print(f"Task: {task}")

    task_result = await task
    print(task_result)

if __name__ == "__main__":
    asyncio.run(main())

 