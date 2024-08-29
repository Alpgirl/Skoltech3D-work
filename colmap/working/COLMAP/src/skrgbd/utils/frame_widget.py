import io

from ipywidgets import Image
import matplotlib.pyplot as plt


class FrameWidget:
    def __init__(self, get_frame, figsize, ticks=True, frames=True, name=False, **kwargs):
        self.get_frame = get_frame
        self.frames = frames
        self._frame_i = 0
        self.name = name

        self.figure, ax = plt.subplots(1, 1, figsize=figsize)
        self.plot = ax.imshow(get_frame(), **kwargs)
        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        self.set_title()
        self.figure.tight_layout()
        plt.close()

        self.buffer = io.BytesIO()
        self.image = Image(value=self.buffer.read(), format='png')

    def update(self):
        self.plot.set_data(self.get_frame())
        self._frame_i += 1
        self.set_title()
        self.buffer.seek(0)
        self.figure.savefig(self.buffer, format='png')
        self.buffer.seek(0)
        self.image.value = self.buffer.read()

    def set_title(self):
        if isinstance(self.name, str) and self.frames:
            title = f'{self.name}'
            if self.frames:
                title = title + f': {self._frame_i}'
            self.figure.suptitle(title)
        else:
            if self.frames:
                title = f'{self._frame_i}'
                self.figure.suptitle(title)
