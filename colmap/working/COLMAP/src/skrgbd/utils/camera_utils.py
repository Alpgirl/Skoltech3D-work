from inspect import cleandoc

from ipywidgets import Box, Layout
from IPython.display import clear_output, display

from skrgbd.utils.parallel import ThreadSet, PropagatingThread as Thread


def auto_white_balance(cameras, figsize_base=5):
    def start_streaming():
        images = []
        for camera in cameras:
            images.append(camera.start_streaming('image', figsize_base=figsize_base, ticks=False))
        for image in images:
            image.width = '220px'
            image.layout.object_fit = 'contain'
        widget = Box(images, layout=Layout(display='flex', flex_flow='row wrap'))
        display(widget)

    def stop_streaming():
        for camera in cameras:
            camera.stop_streaming('image')

    start_streaming()
    input(cleandoc(r"""
    Зафиксируйте белый объект в поле зрения камер так чтобы он занимал максимум поля обзора.
    Нажмите Enter для того чтобы автоматически подобрать баланс белого."""))
    stop_streaming()
    clear_output()

    print('Подбираем баланс белого...')
    threads = ThreadSet([Thread(target=camera.auto_white_balance) for camera in cameras])
    threads.start_and_join()
    start_streaming()
    input('Удостоверьтесь в правильности баланса белого.')
    stop_streaming()
    clear_output()
