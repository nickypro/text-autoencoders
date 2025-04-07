
import time
import concurrent.futures

def exponential_backoff(func, max_retries=10, initial_delay=1, backoff_factor=2):
    """
    A wrapper function to implement exponential backoff for retries.
    """
    def wrapper(*args, **kwargs):
        retries = 0
        delay = initial_delay
        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries == max_retries:
                    print(f"Max retries reached. Error: {e}")
                    raise
                print(f"Retry {retries}/{max_retries}. Waiting {delay} seconds... {e}")
                time.sleep(delay)
                delay *= backoff_factor  # Exponential increase in delay
    return wrapper

def process_in_parallel(items, process_func, max_workers=10):
    """
    Pure parallel processing function.
    Takes a list of items and a processing function.
    Returns a list of results in the same order as input.
    """
    results = [None] * len(items)

    def process_item(item, index):
        try:
            return index, process_func(item)
        except Exception as e:
            print(f"Error processing item {index}: {e}")
            return index, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item, idx) for idx, item in enumerate(items)]

        for future in concurrent.futures.as_completed(futures):
            idx, result = future.result()
            if result is not None:
                results[idx] = result

    return results
