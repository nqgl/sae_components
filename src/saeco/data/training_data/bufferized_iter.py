def bufferized_iter(it, queue_size=32, getnext=next):
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
