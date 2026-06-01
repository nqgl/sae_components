from collections.abc import Callable, Iterator


def bufferized_iter[T](
    it: Iterator[T],
    queue_size: int = 32,
    getnext: Callable[[Iterator[T]], T] = next,
) -> Iterator[T]:
    queue = [getnext(it) for _ in range(queue_size)]

    def qbuf():
        try:
            while True:
                if queue_size == 0:
                    yield getnext(it)
                    continue
                yield queue.pop(0)
                queue.append(getnext(it))
        except StopIteration:
            pass
        yield from queue

    return qbuf()
