def bufferized_iter(it, queue_size=32, getnext=next):
    print("bufferizing", queue_size)
    queue = [getnext(it) for _ in range(queue_size)]
    print("buffer initialized")

    def qbuf():
        try:
            while True:
                yield queue.pop(0)
                queue.append(getnext(it))
        except StopIteration:
            print("buffer depleted")
        for x in queue:
            yield x

    return qbuf()
