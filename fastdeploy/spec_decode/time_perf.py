import time
import paddle

class TimeIntervalRingBufferPaddle:
    def __init__(self, capacity=100):
        self.capacity = capacity
        # buffer: [head, tail, duration0, duration1, ..., durationN]
        self.buffer = paddle.zeros([2 + capacity], dtype='float32').cpu()
        self.last_time = None

    def _increment(self, idx):
        return (idx + 1) % self.capacity

    def add_timestamp(self):
        now = time.time()
        if self.last_time is not None:
            duration = now - self.last_time

            head = int(self.buffer[0].item())
            tail = int(self.buffer[1].item())

            self.buffer[2 + tail] = duration
            tail = self._increment(tail)

            if tail == head:
                head = self._increment(head)

            self.buffer[0] = float(head)
            self.buffer[1] = float(tail)

        self.last_time = now

    def get_durations(self):
        head = int(self.buffer[0].item())
        tail = int(self.buffer[1].item())
        durations = []

        idx = head
        while idx != tail:
            durations.append(float(self.buffer[2 + idx].item()))
            idx = self._increment(idx)
        return durations

    def size(self):
        head = int(self.buffer[0].item())
        tail = int(self.buffer[1].item())
        if tail >= head:
            return tail - head
        else:
            return self.capacity - (head - tail)

    def clear(self):
        self.buffer[:] = 0.0
        self.last_time = None


if __name__ == "__main__":
    buf = TimeIntervalRingBufferPaddle(capacity=5)

    for _ in range(7):
        buf.add_timestamp()
        # time.sleep(0.1)

    print("Size:", buf.size())
    print("Durations:", buf.get_durations())